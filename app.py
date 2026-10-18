import hashlib
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# ---------------- ตั้งค่า ----------------
MODEL_PATH = ("models/fhweights.pt/")
APP_NAME = "Counter"

# สีกรอบของแต่ละชนิด (BGR) ใช้วนไปตามลำดับ
COLORS = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (0, 200, 255), (255, 0, 255)]

st.set_page_config(page_title=APP_NAME, page_icon="🔬", layout="wide")


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


@st.cache_resource
def visitor_log():
    """เก็บชื่อคนที่เข้าใช้ไว้ในหน่วยความจำของเซิร์ฟเวอร์
    (ไม่เขียนลงไฟล์ ปิดเซิร์ฟเวอร์แล้วหายหมด)"""
    return set()


def read_image_bgr(file_bytes: bytes, max_side: int):
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")  # แก้ภาพหมุนจากมือถือ
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------- หาถาด ----------------
def detect_tray_roi(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (180, 80, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box, best_area = None, 0
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area > w * h * 0.25 and cw > w * 0.55 and ch > h * 0.45 and area > best_area:
            best_area = area
            best_box = (x, y, x + cw, y + ch)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    mx, my = int((x2 - x1) * 0.03), int((y2 - y1) * 0.03)
    return x1 + mx, y1 + my, x2 - mx, y2 - my


# ---------------- ตรวจจับแบบแบ่ง 4 ช่อง + NMS + ตัดกรอบซ้อน ----------------
def inside(a, b):
    """สัดส่วนพื้นที่ของกรอบ a ที่อยู่ในกรอบ b"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def detect_objects(model, roi, object_types, conf, imgsz, min_wh, overlap=100, iou=0.4, show_conf=False):
    """คืนค่า (dict จำนวนแต่ละชนิด, ภาพที่วาดกรอบแล้ว)"""
    h, w = roi.shape[:2]
    display = roi.copy()
    hh, hw = h // 2, w // 2
    o = overlap
    x_mid = max(hw - o, 0)
    y_mid = max(hh - o, 0)

    tiles = [
        (roi[0:hh + o, 0:hw + o], 0, 0),
        (roi[0:hh + o, x_mid:w], x_mid, 0),
        (roi[y_mid:h, 0:hw + o], 0, y_mid),
        (roi[y_mid:h, x_mid:w], x_mid, y_mid),
    ]

    boxes, scores, labels = [], [], []
    for tile, off_x, off_y in tiles:
        if tile.size == 0:
            continue
        result = model(tile, conf=conf, iou=iou, imgsz=imgsz, max_det=1000, verbose=False)[0]
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            if name not in object_types:  # เอาเฉพาะชนิดที่เลือก
                continue
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            if (bx2 - bx1) < min_wh or (by2 - by1) < min_wh:
                continue
            boxes.append([int(bx1 + off_x), int(by1 + off_y), int(bx2 + off_x), int(by2 + off_y)])
            scores.append(float(box.conf[0]))
            labels.append(name)

    counts = {name: 0 for name in object_types}

    # NMS แยกทีละชนิด แล้วตัดกรอบที่ซ้อนอยู่ในกรอบที่มั่นใจกว่าเกือบทั้งกรอบ
    fs = max(0.5, w / 1600)   # ขนาดตัวเลขความมั่นใจ ปรับตามขนาดภาพ
    for color_idx, name in enumerate(object_types):
        idx = [i for i, label in enumerate(labels) if label == name]
        if not idx:
            continue
        nms_boxes = [[boxes[i][0], boxes[i][1],
                      boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]] for i in idx]
        keep = [int(k) for k in np.array(cv2.dnn.NMSBoxes(nms_boxes, [scores[i] for i in idx], 0.0, iou)).flatten()]

        kept = []   # เรียงตามความมั่นใจ กรอบที่ซ้อนในกรอบที่มั่นใจกว่าเกิน 70% จะถูกตัด
        for k in sorted(keep, key=lambda k: scores[idx[k]], reverse=True):
            b = boxes[idx[k]]
            if all(max(inside(b, boxes[idx[j]]), inside(boxes[idx[j]], b)) < 0.7 for j in kept):
                kept.append(k)

        color = COLORS[color_idx % len(COLORS)]
        for k in kept:
            bx1, by1, bx2, by2 = boxes[idx[k]]
            cv2.rectangle(display, (bx1, by1), (bx2, by2), color, 2)
            if show_conf:
                label = f"{scores[idx[k]]:.2f}"
                org = (bx1, max(14, by1 - 4))
                cv2.putText(display, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 3)
                cv2.putText(display, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1)
            counts[name] += 1

    return counts, display


@st.cache_data(show_spinner="กำลังนับ...")
def run_pipeline(file_bytes, object_types, conf, imgsz, min_wh, max_side, use_tray, show_conf):
    model = load_model(MODEL_PATH)
    frame = read_image_bgr(file_bytes, max_side)

    tray = detect_tray_roi(frame) if use_tray else None
    tray_found = tray is not None
    if tray is None:
        tray = (0, 0, frame.shape[1], frame.shape[0])  # หาถาดไม่เจอ = นับทั้งภาพ
    x1, y1, x2, y2 = tray

    counts, annotated_roi = detect_objects(
        model, frame[y1:y2, x1:x2].copy(), object_types, conf, imgsz, min_wh, show_conf=show_conf
    )
    out = frame.copy()
    out[y1:y2, x1:x2] = annotated_roi
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), counts, tray_found


# =========================================================
# หน้าแรก: ใส่ชื่อผู้ใช้
# =========================================================
if "user" not in st.session_state:
    st.title(f"🔬 {APP_NAME}")
    st.write("ถ่ายภาพหรืออัปโหลดรูป แล้วให้ระบบช่วยนับจำนวนให้")

    with st.form("login"):
        name = st.text_input("ชื่อผู้ใช้", placeholder="พิมพ์ชื่อของคุณ")
        go = st.form_submit_button("เข้าใช้งาน", type="primary")

    if go:
        name = name.strip()
        if name:
            st.session_state["user"] = name
            visitor_log().add(name)
            print(f"[เข้าใช้งาน] {name}", flush=True)  # ดูได้ในหน้าต่างรัน (Terminal)
            st.rerun()
        else:
            st.warning("กรุณาใส่ชื่อก่อนเข้าใช้งาน")
    st.stop()


# =========================================================
# หน้านับ
# =========================================================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"โหลดโมเดลไม่ได้ ({MODEL_PATH}): {e}")
    st.stop()

with st.sidebar:
    st.write(f"👤 สวัสดี **{st.session_state['user']}**")
    st.caption(f"เข้าใช้งานแล้ว {len(visitor_log())} คน (นับตั้งแต่เปิดเซิร์ฟเวอร์)")
    if st.button("เปลี่ยนผู้ใช้"):
        del st.session_state["user"]
        st.rerun()

   
    with st.expander("⚙️ ตั้งค่าขั้นสูง"):
        conf = st.slider(
            "ระดับความมั่นใจในการตรวจจับ",
            0.01, 0.90, 0.05, 0.01,
            help="ค่ายิ่งสูง ระบบยิ่งคัดกรองเข้มงวด แต่อาจตรวจจับปลาบางตัวไม่พบ"
        )

        imgsz = st.select_slider(
            "ความละเอียดในการตรวจจับ",
            options=[640, 1024, 1280],
            value=1024,
            help="ค่าสูงช่วยให้เห็นรายละเอียดมากขึ้น แต่อาจใช้เวลานับนานขึ้น"
        )

        min_wh = st.number_input(
            "ขนาดกรอบตรวจจับขั้นต่ำ (พิกเซล)",
            min_value=0, value=10, step=1,
            help="กรองกรอบที่เล็กกว่าค่าที่กำหนด ตั้งเป็น 0 เพื่อไม่กรองด้วยขนาด"
        )

        max_side = st.number_input(
            "ขนาดภาพสูงสุดก่อนประมวลผล (พิกเซล)",
            min_value=0, value=1920, step=100,
            help="ระบบจะย่อภาพให้ด้านที่ยาวที่สุดไม่เกินค่านี้ ตั้งเป็น 0 เพื่อไม่ย่อภาพ"
        )

        use_tray = st.checkbox(
            "ตรวจจับเฉพาะบริเวณถาดอัตโนมัติ",
            value=True,
            help="เปิดเพื่อให้ระบบค้นหาถาดและนับเฉพาะในบริเวณถาด หากหาไม่พบจะนับทั้งภาพ"
        )

        show_conf = st.checkbox(
            "แสดงคะแนนความมั่นใจบนกรอบปลา",
            value=False,
            help="เปิดเพื่อแสดงคะแนนความมั่นใจของปลาแต่ละตัวบนภาพผลลัพธ์"
        )

st.title(f"🔬 {APP_NAME}")
st.caption("นับวัตถุด้วย YOLO · ระบบอยู่ระหว่างพัฒนา อาจนับคลาดเคลื่อนเล็กน้อย โปรดตรวจทานก่อนใช้ผล")

# ชนิดที่ให้เลือก = คลาสทั้งหมดที่โมเดลรู้จัก
options = list(dict.fromkeys(model.names.values()))
object_types = st.multiselect("เลือกสิ่งที่จะนับ", options, default=options)
if not object_types:
    st.info("เลือกอย่างน้อย 1 อย่างก่อน")
    st.stop()

tab_cam, tab_up = st.tabs(["📷 ถ่ายด้วยกล้อง", "🖼️ อัปโหลดรูป"])
with tab_cam:
    shot = st.camera_input("ถ่ายภาพ (กดอนุญาตใช้กล้องในเบราว์เซอร์)")
with tab_up:
    uploaded = st.file_uploader("เลือกรูป", type=["jpg", "jpeg", "png"])

source = shot or uploaded
if source is None:
    st.info("ถ่ายภาพหรืออัปโหลดรูปเพื่อเริ่มนับ")
    st.stop()

data = source.getvalue()
img_id = hashlib.md5(data).hexdigest()
if st.session_state.get("img_id") != img_id:  # รูปใหม่ = ล้างค่าปรับแก้
    st.session_state["img_id"] = img_id
    st.session_state["adj"] = 0

annotated, counts, tray_found = run_pipeline(
    data, tuple(object_types), conf, imgsz, int(min_wh), int(max_side), use_tray, show_conf
)

if use_tray and not tray_found:
    st.warning("หาขอบถาดไม่เจอ จึงนับทั้งภาพแทน ลองถ่ายให้เห็นถาดทั้งใบและพื้นหลังต่างสีกับถาด")

st.image(annotated, caption="ผลตรวจจับ (กรอบสี = วัตถุที่นับ, กรอบเขียว = พื้นที่ถาด)", use_container_width=True)

# ---------------- สรุปผล ----------------
adj = st.session_state["adj"]
total = sum(counts.values()) + adj

st.subheader("ผลการนับ")
cols = st.columns(len(counts) + 1)
for col, (name, c) in zip(cols, counts.items()):
    col.metric(name, c)
cols[-1].metric("รวมทั้งหมด", total, delta=f"{adj:+d}" if adj else None, delta_color="off")

st.caption("ถ้าระบบนับพลาด กดปรับเพิ่ม/ลดจำนวนรวมได้")
b1, b2, b3 = st.columns(3)
if b1.button("➖ ลด 1", use_container_width=True):
    st.session_state["adj"] -= 1
    st.rerun()
if b2.button("➕ เพิ่ม 1", use_container_width=True):
    st.session_state["adj"] += 1
    st.rerun()
if b3.button("รีเซ็ต", use_container_width=True):
    st.session_state["adj"] = 0
    st.rerun()

# ---------------- บันทึกรูป (ไม่เก็บข้อมูลอย่างอื่น) ----------------
buf = io.BytesIO()
Image.fromarray(annotated).save(buf, format="PNG")
st.download_button("💾 บันทึกรูปผลตรวจจับ", buf.getvalue(), file_name="result.png", mime="image/png")
