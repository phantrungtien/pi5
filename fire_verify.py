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

YOLO chop tat / vat di chuyen thi doi so hieu vet: vet moi o gan vet thay trong RECHECK s (xem _gan) dung CHUNG
o trang thai, khong hoi lai, khong giu lai lan nua. O di theo hop moi nhat cua vet. Video CHI ve vet duoc di (cung luat voi ROS): nhan AI:OK, hoac YOLO
khi Gemini khong tra loi kip / mat mang. Vet dang hoi hay bi bac khong hien (_FireSaver van luu ca hai).

    python3 fire_verify.py                  # tu kiem logic, khong goi mang
    python3 fire_verify.py KEY a.jpg ...    # hoi Gemini that tung anh
"""
import base64, json, queue, sys, threading, time

import cv2
import requests

from fire_tracker import iou

MODEL = "gemini-3.1-flash-lite"
PROMPT = ("This image is a crop around a region that the onboard fire detector of a firefighting drone flagged "
          "as fire. Is there real fire (visible flames) in it? Smoke alone, sunlight, lamps, reflections and "
          "orange/red objects are NOT fire.")
SCHEMA = {"type": "OBJECT", "properties": {"fire": {"type": "BOOLEAN"}}, "required": ["fire"]}
HOLD = 4.0      # ponytail: cho Gemini toi da 4 s (p90 2 s qua Wi-Fi lab). Bay bang 4G thi do lai.
DOWN = 10.0     # loi mang/API -> bo qua Gemini 10 s roi thu lai
RECHECK = 15.0  # vet bi bac 2 lan lien -> hoi lai sau 15 s: lua that bi bac nham khong bi giu mai
RECHECK1 = 3.0  # bi bac LAN DAU -> hoi lai sau 3 s voi khung khac. 24/10 15:32: anh lua nghieng + nhoe luc
                # vua gio len bi bac, cac vet sau thua ke -> lua bi an 19 s. Bao nham ton them 1 lan goi.
SIDE = 384      # canh dai anh gui; anh cat ~1180 token/lan


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


def _di(s, now):
    """O trang thai -> vet duoc di ra ROS va hien len video. Mot luat cho ca hai: thay gi tren video la
    drone dang nham toi do."""
    return s[0] in ("ok", "yolo") or (s[0] == "?" and now - s[1] >= HOLD)


class FireVerify:
    def __init__(self, key, model=MODEL):
        self.model, self.sa, self.api_key = model, None, None
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
        threading.Thread(target=self._worker, daemon=True).start()
        print(f"[ai] Gemini {model} xac nhan vet lua (giu toi {HOLD:g} s)", flush=True)

    def filter(self, img, tracks, now=None):
        """Vet da xac nhan cua YOLO -> vet duoc phep ra ROS. Goi moi khung tu luong lua."""
        now = time.monotonic() if now is None else now
        out = []
        for t in tracks:
            self.seen[t.id] = now
            if _di(self._state(img, t, now), now):
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
        if s is None or not _di(s, now):
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
                fire = self.ask(jpg.tobytes())
                s[4] = 0 if fire else s[4] + 1
                s[1], s[0] = time.monotonic(), ("ok" if fire else "khong")
                print(f"[ai] hop {s[2]}: {'LUA' if fire else 'khong phai lua'} ({time.monotonic() - t0:.1f} s)",
                      flush=True)
            except Exception as e:  # noqa: BLE001 - mang/API/JSON la: chi YOLO, khong giet luong
                self.down = time.monotonic() + DOWN
                s[1], s[0] = time.monotonic(), "yolo"
                print(f"[ai] loi {type(e).__name__}: {str(e)[:120]} -> chi YOLO {DOWN:g} s", flush=True)

    def ask(self, jpg):
        """JPEG bytes -> True neu Gemini thay lua that."""
        gc = {"responseMimeType": "application/json", "responseSchema": SCHEMA, "maxOutputTokens": 256,
              "thinkingConfig": {"thinkingLevel": "MINIMAL"}}
        body = {"contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpg).decode()}}, {"text": PROMPT}]}],
            "generationConfig": gc}
        if self.sa:
            url = (f"https://aiplatform.googleapis.com/v1/projects/{self.sa['project_id']}/locations/global/"
                   f"publishers/google/models/{self.model}:generateContent")
            h = {"Authorization": f"Bearer {self._token()}"}
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            h = {"x-goog-api-key": self.api_key}
        r = requests.post(url, headers=h, json=body, timeout=(3, 8))
        r.raise_for_status()
        return bool(json.loads(r.json()["candidates"][0]["content"]["parts"][-1]["text"])["fire"])

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
        return a

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
    gate.set()
    wait(fv.v[1], "ok")
    assert fv.filter(img, [a]) == [a] and fv.tag(a) == " AI:OK"

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
    print("OK")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _selfcheck()
    else:
        fv = FireVerify(sys.argv[1])
        for f in sys.argv[2:]:
            im = cv2.imread(f)
            t0 = time.monotonic()
            ok = fv.ask(cv2.imencode(".jpg", crop(im, (0, 0, im.shape[1], im.shape[0])))[1].tobytes())
            print(f"{f}: {'LUA' if ok else 'khong'} ({time.monotonic() - t0:.2f} s)")
