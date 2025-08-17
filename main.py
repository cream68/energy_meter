"""
Streamlit: Energy Meter Reader with Cropping (streamlit-cropper)

Features
- Upload an image (JPG/PNG/HEIC/WEBP) of your meter.
- Crop the area showing the meter ID (OCR check).
- Crop the area showing the meter reading digits (OCR suggestion, manual override).
- Saves readings into CSV + image folder, avoiding duplicates.
- Displays a database and usage charts.

Run:
    streamlit run streamlit_meter_reader_cropper.py

Dependencies:
    pip install streamlit pillow pillow-heif pytesseract pandas numpy streamlit-cropper
"""

import io
import os
import hashlib
from datetime import datetime, date
from typing import Optional

import plotly.graph_objects as go
import cv2
import pandas as pd
import pillow_heif
import numpy as np
from PIL import Image, ExifTags, ImageOps
import streamlit as st
from streamlit_cropper import st_cropper
import json
from pathlib import Path

# OCR
import shutil, pytesseract

TES_BIN = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TES_BIN

# Optional: show status in the UI
def tesseract_status():
    try:
        v = pytesseract.get_tesseract_version()
        return f"Tesseract OK: {TES_BIN} (v{v})"
    except Exception as e:
        return f"Tesseract not available: {e}"

APP_TITLE = "Energy Meter"
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "readings.csv")
METER_ID = "21302018"

os.makedirs(IMAGE_DIR, exist_ok=True)

# Register HEIC/HEIF support
try:
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ------------------
# Helpers
# ------------------

def _image_bytes_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def exif_rgb(img):
    # Use EXIF-corrected orientation everywhere
    
    return ImageOps.exif_transpose(img.convert("RGB"))

def _extract_box_fields(d: dict):
    # accept both left/top and x/y; width/height or w/h
    left = float(d.get("left", d.get("x", 0)))
    top = float(d.get("top", d.get("y", 0)))
    width = float(d.get("width", d.get("w", 0)))
    height = float(d.get("height", d.get("h", 0)))
    return left, top, width, height

def to_norm_payload(box_dict: dict, img_w: int, img_h: int) -> dict:
    """Return normalized (0..1) crop payload."""
    left, top, width, height = _extract_box_fields(box_dict)
    return {
        "norm": True,
        "l": left / img_w,
        "t": top / img_h,
        "w": width / img_w,
        "h": height / img_h,
    }

def preprocess_for_ocr(pil_img):
    # Convert to grayscale
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    # Increase contrast with threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Optional: invert if digits are white on black
    img = cv2.bitwise_not(img)
    # Resize (improves recognition on small digits)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(img)

def payload_to_pixel_box(img, payload: str | dict):
    """
    Convert a saved payload into a Pillow pixel box (left, top, right, bottom)
    on the EXIF-transposed 'img'. Supports:
      - normalized payload: {"norm": True, l,t,w,h}
      - coords wrapper: {"coords": {...}, "ref_width":..., "ref_height":...}
      - raw pixel payload: {"left":..,"top":..,"width":..,"height":..}
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    # unwrap "coords" if present
    if isinstance(data, dict) and "coords" in data:
        box = data["coords"]
    else:
        box = data

    W, H = img.size

    if isinstance(box, dict) and box.get("norm") is True:
        l = int(round(box["l"] * W))
        t = int(round(box["t"] * H))
        r = int(round((box["l"] + box["w"]) * W))
        b = int(round((box["t"] + box["h"]) * H))
    elif "ref_width" in data and "ref_height" in data:
        # scale from reference size to current
        sx = W / float(data["ref_width"])
        sy = H / float(data["ref_height"])
        left, top, width, height = _extract_box_fields(box)
        l = int(round(left * sx))
        t = int(round(top * sy))
        r = int(round((left + width) * sx))
        b = int(round((top + height) * sy))
    else:
        # assume pixel coords on the same image
        left, top, width, height = _extract_box_fields(box)
        l, t, r, b = int(left), int(top), int(left + width), int(top + height)

    # clamp & ensure at least 1px
    l = max(0, min(l, W - 1))
    t = max(0, min(t, H - 1))
    r = max(l + 1, min(r, W))
    b = max(t + 1, min(b, H))
    return (l, t, r, b)


def _parse_exif_dt(img: Image.Image) -> Optional[datetime]:
    try:
        exif = img.getexif()
        if not exif:
            return None
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            if key in tag_map and isinstance(tag_map[key], str):
                exif_dt = tag_map[key].replace("/", ":")
                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                    try:
                        return datetime.strptime(exif_dt, fmt)
                    except Exception:
                        continue
    except Exception:
        return None
    return None

def _load_db() -> pd.DataFrame:
    if os.path.exists(DB_PATH):
        try:
            return pd.read_csv(DB_PATH, parse_dates=["timestamp"], dtype={"meter_id": str})
        except Exception:
            pass
    return pd.DataFrame(columns=["timestamp", "meter_id", "reading_kwh", "image_path", "image_hash", "source_filename"])

def _save_db(df: pd.DataFrame) -> None:
    df.sort_values("timestamp").to_csv(DB_PATH, index=False)

# ------------------
# UI
# ------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    uploaded = st.file_uploader("Upload meter photo", type=["jpg", "jpeg", "png", "heic", "heif", "webp"])
    save_button = st.button("Save Reading", type="primary")

chart_tab, database_tab, edit_tab, ocr_tab = st.tabs(["Usage & Charts", "Database", "Edit Database", "Crop & OCR"])

with ocr_tab:
    if "meter_id_found" not in st.session_state:
        st.session_state.meter_id_found = False
    if uploaded:
        file_bytes = uploaded.read()
        img_hash = _image_bytes_hash(file_bytes)
        uploaded.seek(0)

        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            pil_img = exif_rgb(pil_img)
        except Exception as e:
            st.error(f"Failed to open image: {e}")
            pil_img = None

        if pil_img:
            with st.expander("Image preview", expanded=False):
                st.image(pil_img, caption="Uploaded image preview", use_container_width=True)

            exif_dt = _parse_exif_dt(pil_img) or datetime.now()
            st.success(f"Timestamp: {exif_dt}")

            # --- Crop Meter ID ---
            with st.expander("Step 1: Crop Meter ID area", expanded= not st.session_state.meter_id_found):
                id_box = st_cropper(pil_img, box_color="red", aspect_ratio=None,
                                    return_type="box", key="id_crop")

                id_payload, id_ok = "", False
                if id_box:
                    box = payload_to_pixel_box(pil_img, id_box)
                    cropped_id = pil_img.crop(box)
                    st.image(cropped_id, caption="Cropped Meter ID")

                    if pytesseract:
                        id_text = pytesseract.image_to_string(cropped_id, config="--psm 7 digits").strip()
                        st.write("OCR (ID):", id_text)
                        meter_id_found = METER_ID in "".join(ch for ch in id_text if ch.isdigit())
                        if meter_id_found:
                            id_ok = True
                            st.session_state.meter_id_found = True
                        else:
                            st.warning("Meter ID not matched, but you can continue.")

                    id_payload = json.dumps(to_norm_payload(id_box, pil_img.width, pil_img.height))
            if st.session_state.meter_id_found:
                st.success(f"Meter ID {METER_ID} verified.")
                # --- Crop Reading ---
            with st.expander("Step 2: Crop Meter ID area", expanded= st.session_state.meter_id_found):
                read_box = st_cropper(pil_img, box_color="blue", aspect_ratio=None,
                                    return_type="box", key="reading_crop")

                manual_value, reading_payload = "", ""
                if read_box:
                    box = payload_to_pixel_box(pil_img, read_box)
                    cropped_read = pil_img.crop(box)
                    st.image(cropped_read, caption="Cropped Reading")

                    read_text = ""
                    if pytesseract:
                        proc = preprocess_for_ocr(cropped_read)
                        read_text = pytesseract.image_to_string(cropped_read, config="--psm 7 digits").strip()
                        st.write("OCR (Reading):", read_text)

                    # Get last reading before current timestamp
                    df = _load_db()
                    prev_reading = None
                    if not df.empty:
                        df = df[df["meter_id"] == METER_ID].sort_values("timestamp")
                        before = df[df["timestamp"] < exif_dt]
                        if not before.empty:
                            prev_row = before.iloc[-1]
                            prev_reading = prev_row["reading_kwh"]
                            prev_ts = prev_row["timestamp"]

                    manual_value = st.text_input("Enter/correct meter reading (kWh)")
                    # If we found a previous reading, display it
                    if prev_reading is not None:
                        st.caption(f"📖 Last recorded reading before {exif_dt.strftime('%Y-%m-%d %H:%M')} was **{prev_reading} kWh** at at {prev_ts.strftime('%Y-%m-%d %H:%M')}")

                    reading_payload = json.dumps(to_norm_payload(read_box, pil_img.width, pil_img.height))

                # --- Save ---
                if save_button and manual_value:
                    try:
                        reading_val = float(manual_value)
                    except ValueError:
                        st.error("Invalid manual value.")
                        reading_val = None

                    if reading_val is not None:
                        safe_date = exif_dt.strftime("%Y-%m-%d_%H%M%S")
                        dest_name = f"{safe_date}_{METER_ID}.jpg"
                        dest_path = os.path.join(IMAGE_DIR, dest_name)
                        pil_img.save(dest_path, "JPEG", quality=92)

                        df = _load_db()
                        if not (
                            ((df["meter_id"] == METER_ID) & (df["timestamp"] == pd.Timestamp(exif_dt))).any()
                            or (df["image_hash"] == img_hash).any()
                        ):
                            new_row = {
                                "timestamp": exif_dt,
                                "meter_id": METER_ID,
                                "reading_kwh": reading_val,
                                "image_path": dest_path,
                                "image_hash": img_hash,
                                "source_filename": uploaded.name,
                                "id_crop_box": id_payload,
                                "reading_crop_box": reading_payload,
                            }
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            _save_db(df)
                            st.success(f"Saved reading: {reading_val} kWh at {exif_dt}")
                        else:
                            st.warning("Duplicate entry detected, skipping save.")

with database_tab:
    st.subheader("Database")
    df = _load_db()
    if df.empty:
        st.info("No readings yet.")
    else:
        df_show = df.copy()

        # keep only useful columns
        cols = ["timestamp", "meter_id", "reading_kwh", "image_path"]
        st.dataframe(df_show[cols], use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "readings.csv",
            "text/csv",
        )

        with st.expander("Bildvorschau", expanded=False):

            # Select one row to preview the image
            options = [f"{i}: {row['timestamp']} | {row['meter_id']} | {row['reading_kwh']} kWh"
                    for i, row in df_show.iterrows()]
            choice = st.selectbox("Preview image for row:", options)

            if choice:
                idx = int(choice.split(":")[0])
                row = df_show.loc[idx]
                img_path = row["image_path"]

                if isinstance(img_path, str) and os.path.exists(img_path):
                    # Preview
                    st.image(img_path, caption=f"Image for {row['timestamp']}")

                    # Download button
                    with open(img_path, "rb") as f:
                        st.download_button(
                            label="📥 Download image",
                            data=f,
                            file_name=os.path.basename(img_path),
                            mime="image/jpeg",
                        )
                else:
                    st.warning("Image file not found.")

with edit_tab:
    st.subheader("Edit or Remove Readings")
    df = _load_db()

    if df.empty:
        st.info("No readings available.")
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
        options = [f"{i}: {row['timestamp']} | {row['meter_id']} | {row['reading_kwh']} kWh"
                   for i, row in df.iterrows()]
        choice = st.selectbox("Select a row to edit", options)

        if choice:
            idx = int(choice.split(":")[0])
            row = df.loc[idx]

            if isinstance(row["image_path"], str) and os.path.exists(row["image_path"]):
                full_img = exif_rgb(Image.open(row["image_path"]))

                # Show cropped ID
                if row.get("id_crop_box"):
                    try:
                        box = payload_to_pixel_box(full_img, row["id_crop_box"])
                        st.image(full_img.crop(box), caption="Cropped Meter ID", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not crop ID: {e}")

                new_meter_id = st.text_input("Meter ID", row["meter_id"])

                # Show cropped Reading
                if row.get("reading_crop_box"):
                    try:
                        box = payload_to_pixel_box(full_img, row["reading_crop_box"])
                        st.image(full_img.crop(box), caption="Cropped Meter Reading", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not crop Reading: {e}")

                new_reading = st.number_input("Reading (kWh)", value=float(row["reading_kwh"]))

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Update Row"):
                    df.at[idx, "meter_id"] = new_meter_id
                    df.at[idx, "reading_kwh"] = new_reading
                    _save_db(df)
                    st.success("Row updated successfully.")
            with col2:
                if st.button("🗑️ Remove Row"):
                    df = df.drop(idx).reset_index(drop=True)
                    _save_db(df)
                    st.success("Row removed successfully.")

with chart_tab:
    st.subheader("Usage & Charts")
    df = _load_db()
    if df.empty:
        st.info("No data yet.")
    else:
        df = df[df["meter_id"] == METER_ID].sort_values("timestamp").reset_index(drop=True)
        col1, col2 = st.columns(2)
        min_d = df["timestamp"].min().date()
        with col1:
            start_d = st.date_input("Start date", value=min_d, min_value=min_d, max_value=date.today())
        with col2:
            end_d = st.date_input("End date", value=date.today(), min_value=min_d, max_value=date.today())
            end_dt = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Toggle: Interpolate first value
        interpolate_first = st.checkbox("Interpolate first value at start date", value=False)

        if start_d <= end_d:
            mask = (df["timestamp"] >= pd.Timestamp(start_d)) & (df["timestamp"] <= pd.Timestamp(end_dt))
            dfr = df.loc[mask].copy()

            # Handle interpolation if selected
            if interpolate_first and not dfr.empty:
                # Find last reading before start_d
                prev = df[df["timestamp"] < pd.Timestamp(start_d)].tail(1)
                if not prev.empty:
                    prev_row = prev.iloc[0]
                    first_row = dfr.iloc[0]

                    # Linear interpolation of reading at start_d
                    delta_t = (first_row["timestamp"] - prev_row["timestamp"]).total_seconds()
                    delta_r = first_row["reading_kwh"] - prev_row["reading_kwh"]
                    rate = delta_r / delta_t if delta_t > 0 else 0
                    seconds_from_prev = (pd.Timestamp(start_d) - prev_row["timestamp"]).total_seconds()
                    interpolated_value = prev_row["reading_kwh"] + rate * seconds_from_prev

                    # Insert synthetic row
                    synthetic = first_row.copy()
                    synthetic["timestamp"] = pd.Timestamp(start_d)
                    synthetic["reading_kwh"] = interpolated_value
                    synthetic["image_path"] = (
                        f"Interpolated between {prev_row['timestamp'].date()} "
                        f"({prev_row['reading_kwh']}) and {first_row['timestamp'].date()} "
                        f"({first_row['reading_kwh']})"
                    )
                    dfr = pd.concat([pd.DataFrame([synthetic]), dfr], ignore_index=True)
                    dfr = dfr.sort_values("timestamp").reset_index(drop=True)

            if len(dfr) >= 2:
                dfr["usage_kwh"] = dfr["reading_kwh"].diff()
                dfr["usage_kwh_sum"] = dfr["usage_kwh"].cumsum()
                # ensure first row is zero instead of NaN
                dfr.loc[dfr.index[0], "usage_kwh"] = 0
                dfr.loc[dfr.index[0], "usage_kwh_sum"] = 0

                with st.expander("Datensatz", expanded=False):
                    cols = list(range(0, 4)) + [-2, -1]
                    st.dataframe(dfr.iloc[:, cols])

                col1, col2 = st.columns(2)
                total_usage = dfr["usage_kwh"].sum()
                min_d = dfr["timestamp"].min()
                max_d = dfr["timestamp"].max()
                total_seconds = (max_d - min_d).total_seconds()
                months = total_seconds / (30 * 24 * 3600)
                avg_monthly_usage = total_usage / months if months > 0 else 0
                with col1:
                    st.metric("Total usage (selected)", f"{total_usage:.2f} kWh")
                with col2:
                    st.metric("Avg monthly usage", f"{avg_monthly_usage:.2f} kWh")

                # --- Plot ---
                fig = go.Figure()

                # Step line for usage
                fig.add_trace(go.Scatter(
                    x=dfr["timestamp"],
                    y=dfr["usage_kwh"],
                    name="Usage (kWh per interval)",
                    mode="lines+markers",
                    line=dict(color="blue", width=2, shape="hv"),
                    hovertemplate="Usage: %{y:.0f} kWh",
                ))

                # Line for cumulative usage
                fig.add_trace(go.Scatter(
                    x=dfr["timestamp"],
                    y=dfr["usage_kwh_sum"],
                    name="Cumulative Usage (kWh)",
                    mode="lines+markers",
                    line=dict(color="red", width=2),
                    hovertemplate="Usage: %{y:.0f} kWh",
                ))

                fig.update_layout(
                    title="Energy Usage & Cumulative Usage",
                    xaxis_title="Timestamp",
                    yaxis_title="kWh",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(0,0,0,0)"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 readings in range.")
