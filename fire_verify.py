"""Gemini xac nhan vet lua YOLO (Vertex AI, service account). YOLO van la lop chinh, chay offline.

    run_gui.py --fire-model ... --fire-verify ~/jetson/hidden-lyceum-473116-p8-1f580efa4c6d.json

Vet MOI -> cat anh quanh hop -> luong rieng hoi Gemini "co lua that khong". Do 24/10/2026 tu Jetson
(~/fire_bench, 48 anh cat tu hop YOLO): gemini-3.1-flash-lite dung 48/48, p50 1.62 s / p90 2.01 s,
~$0.31 / 1000 lan (tru credit trial). 3.7/3.8-flash p90 > 20 s, co lan 429 -> khong dung cho drone.

filter() quyet vet nao di tiep ra /fire/detections (fire_locator -> /fire/target -> drone bay toi):

    Gemini noi CO lua          -> di
    Gemini noi KHONG           -> giu (thanh an toan vang cua robot, den, hoang hon...); hoi lai sau
                                  RECHECK1 (lan dau) hoac RECHECK (bi bac 2 lan lien)
    dang hoi, chua qua HOLD    -> giu: fire_locator chi can 5 khung (1 s) la phat /fire/target - cho qua
                                  thi bao nham da keo drone di truoc khi Gemini kip tra loi
    dang hoi, qua HOLD         -> di (chi YOLO): Gemini cham/treo khong duoc bit mat drone
    mat mang / loi API         -> di ngay (chi YOLO) trong DOWN s, khong cho HOLD
    hai dong chi YOLO tren chi di khi diem >= YOLO_ONLY: --fire-conf thap (0.2) la de Gemini thay nhieu
    hon, khong co Gemini thi giu nguong cu

YOLO chop tat / vat di chuyen thi doi so hieu vet: vet moi o gan vet thay trong RECHECK s (xem _gan) dung CHUNG
o trang thai, khong hoi lai, khong giu lai lan nua. O di theo hop moi nhat cua vet. Video CHI ve vet duoc di (cung luat voi ROS): nhan AI:OK, hoac YOLO
khi Gemini khong tra loi kip / mat mang. Vet dang hoi hay bi bac khong hien (_FireSaver van luu ca hai).

    python3 fire_verify.py                  # tu kiem logic, khong goi mang
    python3 fire_verify.py KEY a.jpg ...    # hoi Gemini that tung anh
"""
import base64, json, queue, sys, threading, time
import concurrent.futures as cf

import cv2
import requests

from fire_tracker import iou

# Hoi CA HAI cung luc, LUA chi khi CA HAI noi lua (24/10 18:28 theo y nguoi dung; truoc do lay cau ve truoc):
# anh lua ve/tren man hinh hai model bat dong (3.1 khong, 3.5 lua), lua that ca hai 48/48 tren ~/fire_bench.
# Gemini cham theo dot (24/10 14:10 3.1-flash-lite 5-8 s, cung luc 3.5 p50 1.4 s) -> WAIT2.
MODELS = ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite")
WAIT2 = 3.0     # mot model da noi lua, model kia cham qua WAIT2 s -> nhan mot phieu (log ghi ten mot model).
                # Dang bay: cho them = lua troi khoi khung. Doi 2/2 tuyet doi thi dat 99.
WORKERS = 3     # vet khac nhau hoi song song: anh lua khong xep hang sau bao nham trong lab
PROMPT = ("This image is a crop around a region that the onboard fire detector of a firefighting drone flagged "
          "as fire. Is there real fire (visible flames) in it? Smoke alone, sunlight, lamps, reflections and "
          "orange/red objects are NOT fire.")
SCHEMA = {"type": "OBJECT", "properties": {"fire": {"type": "BOOLEAN"}}, "required": ["fire"]}
HOLD = 4.0      # ponytail: cho Gemini toi da 4 s (p90 2 s qua Wi-Fi lab). Bay bang 4G thi do lai.
DOWN = 10.0     # loi mang/API -> bo qua Gemini 10 s roi thu lai
RECHECK = 15.0  # vet bi bac 2 lan lien -> hoi lai sau 15 s: lua that bi bac nham khong bi giu mai
RECHECK1 = 1.0  # bi bac LAN DAU -> hoi lai sau 1 s voi khung khac. 24/10 15:32 va 15:52: anh lua luc vua gio
                # len (nghieng, nhoe) deu bi bac, lan hai dung. Hoi song song nen hoi lai som khong nghen ai.
YOLO_ONLY = 0.35  # chi YOLO (mat mang/Gemini cham) thi doi diem nhu truoc khi co Gemini. 24/10 18:12 ha
                  # --fire-conf 0.35 -> 0.2: anh dien thoai 6 hinh lua YOLO cho 0.37/0.21/0.13, chi qua 1
SIDE = 384      # canh dai anh gui; anh cat ~1180 token/lan
# Ghi song song (--fire-scan): Gemini quet NGUYEN khung tim moi vung lua, so voi YOLO cung khung, chi ghi log
# + luu khung lech (YOLO sot = anh de train lai). Khong anh huong ROS. 24/10 tren 12 khung anh dien thoai 6 hinh
# lua: 3.1 dung 6/6 o 11/12 khung, YOLO 0.2 chi 1-3; ~1316 token vao + 255 ra = ~$0.0007/lan, p50 2.7 s qua 4G.
SCAN_MODEL = "gemini-3.1-flash-lite"  # 3.5 khoanh them hinh bi cat mep (6-8 hop), dat hon 1.5x
SCAN_PROMPT = ("Find every region of this drone camera frame that shows flames, including flames in photos, "
               "drawings or on screens. For each give box_2d [ymin, xmin, ymax, xmax] normalized to 0-1000, and "
               "real=true only if it is actual fire in the scene (not a picture or a screen). Empty list if none.")
SCAN_SCHEMA = {"type": "OBJECT", "properties": {"fires": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}}, "real": {"type": "BOOLEAN"}},
    "required": ["box_2d", "real"]}}}, "required": ["fires"]}


def crop(img, box):
    """Vuong quanh hop, rong gap 1.5 canh dai hop (Gemini can ngu canh: den tran hay lua), toi thieu 128 px."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    s = max(1.5 * max(x2 - x1, y2 - y1), 128)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    c = img[int(max(0, cy - s / 2)):int(min(h, cy + s / 2)), int(max(0, cx - s / 2)):int(min(w, cx + s / 2))]
    k = SIDE / max(c.shape[:2])
    return cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else c


def _gan(a, b):
    """Hop b co phai cung vat voi hop a (vua thay): chong nhau, hoac tam b cach tam a <= canh dai cua a."""
    if iou(a, b) >= 0.3:
        return True
    s = max(a[2] - a[0], a[3] - a[1])
    return abs(a[0] + a[2] - b[0] - b[2]) / 2 <= s and abs(a[1] + a[3] - b[1] - b[3]) / 2 <= s


def so_khop(gem, yolo):
    """Hop Gemini vs hop YOLO cung khung -> (khop, chi Gemini thay, chi YOLO thay): ba danh sach hop."""
    cung = lambda a, b: _gan(a, b) or _gan(b, a)  # noqa: E731 - hop Gemini rong hon hop YOLO
    khop = [y for y in yolo if any(cung(g, y) for g in gem)]
    return khop, [g for g in gem if not any(cung(g, y) for y in yolo)], [y for y in yolo if y not in khop]


def _di(s, t, now):
    """O trang thai + vet -> vet duoc di ra ROS va hien len video. Mot luat cho ca hai: thay gi tren video la
    drone dang nham toi do."""
    if s[0] == "ok":
        return True
    return t.score >= YOLO_ONLY and (s[0] == "yolo" or (s[0] == "?" and now - s[1] >= HOLD))


class FireVerify:
    def __init__(self, key, models=MODELS, scan_dir=None):
        self.models, self.sa, self.api_key, self.scan_dir = models, None, None, scan_dir
        if key is not None:  # doc khoa NGAY: sai duong dan thi keu luc khoi dong, khong phai luc dang bay
            if key.endswith(".json"):
                self.sa = json.load(open(key))
            else:  # ponytail: file chua API key AI Studio (free tier). Chua thu that - credit trial het 22/01/2027
                self.api_key = open(key).read().strip()
        self.v = {}      # so hieu vet -> o [ket qua, gio, hop]; ket qua "?" | "ok" | "khong" | "yolo"
        self.seen = {}   # so hieu vet -> lan cuoi thay, de don self.v
        self.down = 0.0  # truoc moc nay (monotonic) coi nhu mat mang
        self.q = queue.Queue()
        self._tok, self._exp, self._prune = None, 0.0, 0.0
        self._scan_next, self._scan_busy = 0.0, False
        self.pool = cf.ThreadPoolExecutor(max_workers=WORKERS * len(models))
        for _ in range(WORKERS):
            threading.Thread(target=self._worker, daemon=True).start()
        print(f"[ai] Gemini {' + '.join(models)} cung noi lua moi tinh (giu toi {HOLD:g} s)", flush=True)

    def filter(self, img, tracks, now=None):
        """Vet da xac nhan cua YOLO -> vet duoc phep ra ROS. Goi moi khung tu luong lua."""
        now = time.monotonic() if now is None else now
        out = []
        for t in tracks:
            self.seen[t.id] = now
            if _di(self._state(img, t, now), t, now):
                out.append(t)
        if now - self._prune > 10:
            self._prune = now
            for i in [i for i, ts in self.seen.items() if now - ts > 60]:
                del self.seen[i], self.v[i]
        return out

    def tag(self, t, now=None):
        """Nhan them sau ten lop tren video (FireTracker.draw_cached); None = khong ve vet nay."""
        s = self.v.get(t.id)
        now = time.monotonic() if now is None else now
        if s is None or not _di(s, t, now):
            return None
        return " AI:OK" if s[0] == "ok" else " YOLO"

    def _state(self, img, t, now):
        """o [ket qua, gio hoi/tra loi, hop MOI NHAT, lan cuoi thay] cua vet t; vet moi thi hoi Gemini."""
        s = self.v.get(t.id)
        if s is None:
            # YOLO mat dau roi bat lai (vat dang di chuyen) -> so hieu moi: dung chung o cua vet thay trong
            # RECHECK s o gan do. 24/10: o dung yen o cho hoi -> dien thoai cam tay bi hoi lai 9 lan lien;
            # cua so 3 s thi bao nham chop tat (thay 1 khung, mat vai giay) bi hoi lai moi lan hien.
            s = next((o for o in self.v.values() if now - o[3] < RECHECK and _gan(o[2], t.box)), None)
            if s is None:
                s = [None, now, t.box, now, 0]  # o[4]: so lan "khong" lien tiep
            self.v[t.id] = s
        s[2], s[3] = t.box, now  # o di theo vet
        if (s[0] is None or (s[0] == "khong" and now - s[1] > (RECHECK if s[4] >= 2 else RECHECK1))
                or (s[0] == "yolo" and now >= self.down)):
            if now < self.down:
                s[1], s[0] = now, "yolo"
            else:
                s[1], s[0] = now, "?"
                self.q.put((s, cv2.imencode(".jpg", crop(img, t.box), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]))
        return s

    def _worker(self):
        while True:
            s, jpg = self.q.get()
            t0 = time.monotonic()
            try:
                fire, m = self.ask(jpg.tobytes())
                s[4] = 0 if fire else s[4] + 1
                s[1], s[0] = time.monotonic(), ("ok" if fire else "khong")
                print(f"[ai] hop {s[2]}: {'LUA' if fire else 'khong phai lua'} ({time.monotonic() - t0:.1f} s, "
                      f"{m.replace('gemini-', '')})", flush=True)
            except Exception as e:  # noqa: BLE001 - mang/API/JSON la: chi YOLO, khong giet luong
                self.down = time.monotonic() + DOWN
                s[1], s[0] = time.monotonic(), "yolo"
                print(f"[ai] loi {type(e).__name__}: {str(e)[:120]} -> chi YOLO {DOWN:g} s", flush=True)

    def ask(self, jpg):
        """JPEG bytes -> (co lua that?, model). Hoi cac model song song; LUA chi khi moi model tra loi duoc deu
        noi lua. Mot model noi khong -> khong NGAY (khong cho model kia). Model loi thi bo phieu cua no; noi lua
        roi ma model kia cham qua WAIT2 -> nhan mot phieu. Tat ca loi moi nem loi (-> chi YOLO)."""
        futs = {self.pool.submit(self._ask1, m, jpg): m for m in self.models}
        yes, err, end, cho = [], None, None, set(futs)
        while cho:  # model cham van chay tiep o nen, ket qua bo
            xong, cho = cf.wait(cho, timeout=None if end is None else max(0.0, end - time.monotonic()),
                                return_when=cf.FIRST_COMPLETED)
            if not xong:
                break
            for f in xong:
                if f.exception() is not None:
                    err = f.exception()
                elif not f.result():
                    return False, futs[f]
                else:
                    yes.append(futs[f])
                    end = end or time.monotonic() + WAIT2
        if yes:
            return True, "+".join(yes)
        raise err

    def scan(self, img, tracks, every):
        """Ghi song song: moi `every` s gui nguyen khung cho Gemini o luong rieng, so voi hop YOLO (CHUA loc)
        cua chinh khung do. Goi moi khung tu luong lua; khong tra gi, khong dung toi ROS."""
        now = time.monotonic()
        if self._scan_busy or now < self._scan_next:
            return
        self._scan_busy, self._scan_next = True, now + every
        yolo = [(tuple(int(v) for v in t.box), round(float(t.score), 2)) for t in tracks]
        threading.Thread(target=self._scan1, args=(img.copy(), yolo), daemon=True).start()

    def _scan1(self, img, yolo):
        t0, wall = time.monotonic(), time.time()
        try:
            h, w = img.shape[:2]
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            gem = [((int(b[1] * w / 1000), int(b[0] * h / 1000), int(b[3] * w / 1000), int(b[2] * h / 1000)),
                    bool(f["real"])) for f in self._post(SCAN_MODEL, jpg, SCAN_PROMPT, SCAN_SCHEMA, 1024, 15)["fires"]
                   if len(b := f["box_2d"]) == 4]
            dt = time.monotonic() - t0
            khop, chi_g, chi_y = so_khop([b for b, _ in gem], [b for b, _ in yolo])
            that = sum(r for _, r in gem)
            print(f"[scan] Gemini {len(gem)} hop ({that} lua that) | YOLO {len(yolo)} | khop {len(khop)}, YOLO sot "
                  f"{len(chi_g)}, chi YOLO {len(chi_y)} ({dt:.1f} s)", flush=True)
            if self.scan_dir:
                import os
                d = os.path.join(self.scan_dir, time.strftime("%Y%m%d", time.localtime(wall)))
                os.makedirs(d, exist_ok=True)
                ten = None
                if chi_g or chi_y:  # lech -> luu khung: YOLO sot lua la anh can nhat de train lai
                    ten = time.strftime("%H%M%S", time.localtime(wall)) + f"_{int(wall * 1000) % 1000:03d}.jpg"
                    cv2.imwrite(os.path.join(d, ten), img)
                with open(os.path.join(d, "scan.jsonl"), "a") as f:
                    f.write(json.dumps({"t": round(wall, 3), "s": round(dt, 2), "jpg": ten,
                                        "gemini": [list(b) + [r] for b, r in gem],
                                        "yolo": [list(b) + [sc] for b, sc in yolo]}) + "\n")
        except Exception as e:  # noqa: BLE001 - chi la ghi log: loi gi cung bo qua, khong cham toi xac nhan
            print(f"[scan] loi {type(e).__name__}: {str(e)[:120]}", flush=True)
        finally:
            self._scan_busy = False

    def _ask1(self, model, jpg):
        """Mot lan goi mot model -> True neu thay lua that."""
        return bool(self._post(model, jpg, PROMPT, SCHEMA, 256, 8)["fire"])

    def _post(self, model, jpg, prompt, schema, max_out, read_timeout):
        """Anh + cau hoi -> JSON theo schema."""
        # temperature 0: cung anh cung dap an (24/10/2026 mac dinh 1.0 tra T,F,F cho cung mot khung).
        # Google khuyen giu 1.0 cho Gemini 3 (so lap/giam chat luong) - dau ra chi JSON ngan nen thu 0.
        gc = {"responseMimeType": "application/json", "responseSchema": schema, "maxOutputTokens": max_out,
              "temperature": 0, "thinkingConfig": {"thinkingLevel": "MINIMAL"}}
        body = {"contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpg).decode()}}, {"text": prompt}]}],
            "generationConfig": gc}
        if self.sa:
            url = (f"https://aiplatform.googleapis.com/v1/projects/{self.sa['project_id']}/locations/global/"
                   f"publishers/google/models/{model}:generateContent")
            h = {"Authorization": f"Bearer {self._token()}"}
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            h = {"x-goog-api-key": self.api_key}
        r = requests.post(url, headers=h, json=body, timeout=(3, read_timeout))
        r.raise_for_status()
        return json.loads(r.json()["candidates"][0]["content"]["parts"][-1]["text"])

    def _token(self):
        """Service account -> access token (pyjwt + cryptography co san tren Jetson), lam moi truoc 5 phut."""
        if time.time() < self._exp - 300:
            return self._tok
        import jwt
        now = int(time.time())
        a = jwt.encode({"iss": self.sa["client_email"], "scope": "https://www.googleapis.com/auth/cloud-platform",
                        "aud": "https://oauth2.googleapis.com/token", "iat": now, "exp": now + 3600},
                       self.sa["private_key"], algorithm="RS256")
        r = requests.post("https://oauth2.googleapis.com/token", timeout=(3, 8),
                          data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": a})
        r.raise_for_status()
        self._tok, self._exp = r.json()["access_token"], now + r.json().get("expires_in", 3600)
        return self._tok


def _selfcheck():
    import numpy as np
    from fire_tracker import Track

    answers, gate, asked = {}, threading.Event(), []

    def fake_ask(jpg):
        asked.append(1)
        gate.wait(2)
        a = answers.pop("next")
        if isinstance(a, Exception):
            raise a
        return a, "gia"

    def wait(s, want):
        for _ in range(200):
            if s[0] == want:
                return
            time.sleep(0.01)
        raise AssertionError(f"cho {want}, van {s[0]}")

    img = np.zeros((720, 1280, 3), np.uint8)
    fv = FireVerify(None)
    fv.ask = fake_ask
    a = Track(1, (100, 100, 160, 160), 0, 0.6)

    answers["next"] = True
    assert fv.filter(img, [a]) == [], "dang hoi -> giu"
    assert fv.tag(a) is None, "dang hoi -> khong hien len video"
    assert fv.filter(img, [a], now=time.monotonic() + HOLD) == [a], "qua HOLD -> chi YOLO"
    assert fv.tag(a, now=time.monotonic() + HOLD) == " YOLO"
    a.score = 0.25
    assert fv.filter(img, [a], now=time.monotonic() + HOLD) == [], "chi YOLO ma diem < YOLO_ONLY -> giu"
    gate.set()
    wait(fv.v[1], "ok")
    assert fv.filter(img, [a]) == [a] and fv.tag(a) == " AI:OK", "Gemini xac nhan -> diem thap van di"
    a.score = 0.6

    a2 = Track(2, (104, 102, 162, 158), 0, 0.5)  # YOLO chop tat, doi so hieu, cung cho
    assert fv.filter(img, [a2]) == [a2] and len(asked) == 1, "thua ke, khong hoi lai"
    a2.box = (180, 100, 240, 160)  # cam tay di chuyen, cung so hieu: o di theo
    fv.filter(img, [a2])
    a3 = Track(6, (225, 100, 285, 160), 0, 0.5)  # mat dau roi bat lai lech 45 px, IoU 0.14
    assert fv.filter(img, [a3]) == [a3] and len(asked) == 1, "vet moi canh vet vua thay -> thua ke"

    gate.clear()
    b = Track(3, (900, 400, 960, 460), 0, 0.55)  # thanh vang
    answers["next"] = False
    fv.filter(img, [b])
    gate.set()
    wait(fv.v[3], "khong")
    assert fv.filter(img, [b]) == [] and fv.tag(b) is None
    assert fv.filter(img, [a, b]) == [a]

    from fire_tracker import FireTracker  # video ve DUNG vet ROS nhan: a co, b (bi bac) khong
    class _Gia:
        names, ms, hud_label, tracks, tag = {0: "Fire"}, 1.0, "LUA", [a, b], staticmethod(fv.tag)
    ve = np.zeros_like(img)
    FireTracker.draw_cached(_Gia(), ve, hud="")
    assert ve[100:160, 100].any() and not ve[400:460, 900].any(), "vet bi bac khong duoc ve"

    n = len(asked)  # bac lan dau: sau RECHECK1 hoi lai; bac 2 lan lien: im toi RECHECK
    answers["next"] = False
    assert fv.filter(img, [b], now=time.monotonic() + RECHECK1 + 0.5) == [], "dang hoi lai -> van giu"
    for _ in range(200):
        if fv.v[3][4] == 2:
            break
        time.sleep(0.01)
    assert len(asked) == n + 1 and fv.v[3][0] == "khong" and fv.v[3][4] == 2
    fv.filter(img, [b], now=time.monotonic() + RECHECK1 + 0.5)
    assert len(asked) == n + 1, "bac 2 lan lien -> khong hoi lai truoc RECHECK"

    c = Track(4, (500, 500, 540, 540), 0, 0.4)  # mat mang
    answers["next"] = requests.ConnectionError("no route")
    fv.filter(img, [c])
    wait(fv.v[4], "yolo")
    assert fv.filter(img, [c]) == [c] and fv.tag(c) == " YOLO"
    d = Track(5, (300, 600, 340, 640), 0, 0.4)
    n = len(asked)
    assert fv.filter(img, [d]) == [d] and len(asked) == n, "trong DOWN: di ngay, khong hoi"

    global WAIT2
    hai = FireVerify(None, models=("cham", "nhanh"))  # 2/2: ca hai noi lua moi tinh
    tra = {"cham": True, "nhanh": True}
    hai._ask1 = lambda m, jpg: (time.sleep(0.2) if m == "cham" else None, tra[m])[1]
    assert hai.ask(b"") == (True, "nhanh+cham"), "ca hai lua -> lua, cho model cham"
    tra["nhanh"] = False
    t0 = time.monotonic()
    assert hai.ask(b"") == (False, "nhanh") and time.monotonic() - t0 < 0.15, "mot model noi khong -> khong ngay"
    tra.update(nhanh=True, cham=False)
    assert hai.ask(b"") == (False, "cham"), "nhanh noi lua, cham noi khong -> khong"
    WAIT2, tra["cham"] = 0.05, True
    t0 = time.monotonic()
    assert hai.ask(b"") == (True, "nhanh") and time.monotonic() - t0 < 0.15, "model kia cham qua WAIT2 -> 1 phieu"
    WAIT2 = 3.0

    def nhanh_loi(m, jpg):
        if m == "nhanh":
            raise requests.ConnectionError("x")
        return tra[m]
    hai._ask1 = nhanh_loi
    assert hai.ask(b"") == (True, "cham"), "model loi -> lay phieu model kia"
    hai._ask1 = lambda m, jpg: (_ for _ in ()).throw(requests.ConnectionError(m))
    try:
        hai.ask(b"")
        raise AssertionError("ca hai loi phai nem loi")
    except requests.ConnectionError:
        pass

    g = [(100, 100, 200, 220), (300, 100, 400, 200), (600, 50, 700, 150)]  # 3 hinh lua, YOLO thay 1 + 1 bao nham
    y = [(110, 120, 190, 210), (900, 600, 950, 650)]
    assert so_khop(g, y) == ([y[0]], g[1:], [y[1]])
    assert so_khop([], y) == ([], [], y) and so_khop(g, []) == ([], g, [])

    import tempfile
    sc = FireVerify(None, scan_dir=tempfile.mkdtemp())
    sc._post = lambda *a: {"fires": [{"box_2d": [139, 78, 306, 156], "real": True}]}  # 0-1000 -> (100,100,200,220)
    sc.scan(img, [a, c], every=10)
    sc.scan(img, [a], every=10)  # trong `every` s / dang chay: bo qua
    for _ in range(200):
        if not sc._scan_busy:
            break
        time.sleep(0.01)
    import glob as _g
    rec = [json.loads(x) for f in _g.glob(sc.scan_dir + "/*/scan.jsonl") for x in open(f)]
    assert len(rec) == 1 and rec[0]["gemini"] == [[99, 100, 199, 220, True]] and len(rec[0]["yolo"]) == 2
    assert rec[0]["jpg"], "lech (YOLO c khong khop) -> luu khung"
    print("OK")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _selfcheck()
    else:
        fv = FireVerify(sys.argv[1])
        for f in sys.argv[2:]:
            im = cv2.imread(f)
            t0 = time.monotonic()
            ok, m = fv.ask(cv2.imencode(".jpg", crop(im, (0, 0, im.shape[1], im.shape[0])))[1].tobytes())
            print(f"{f}: {'LUA' if ok else 'khong'} ({time.monotonic() - t0:.2f} s, {m})")
