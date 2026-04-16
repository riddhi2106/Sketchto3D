"""
Sketch -> 3D : Streamlit UI (Member 4)
======================================
End-to-end demo wiring together Members 1-4.

Run:
    streamlit run app.py
"""

import io

import numpy as np
import cv2
import streamlit as st
from PIL import Image
from sklearn.datasets import load_digits
from streamlit_drawable_canvas import st_canvas

from preprocess import preprocess_image
from edge_detect import detect_edges
from coords_extract import extract_coordinates
from matrix_member2 import run_member2_pipeline
from matrix_ops import run_member3_pipeline
from extrude import extrude_points, extrude_mesh
from visualize_3d import build_figure, build_mesh_figure


# ─────────────────────────────────────────────
# Image loading helpers
# ─────────────────────────────────────────────
def load_uploaded_image(upload) -> np.ndarray:
    """Decode an uploaded image into a 2D grayscale uint8 array."""
    img = Image.open(upload).convert("L")
    return np.array(img, dtype=np.uint8)


def load_digit_sample(index: int) -> np.ndarray:
    """Fetch one 8x8 MNIST-style digit as a 2D array."""
    digits = load_digits()
    return digits.images[index % len(digits.images)]


def load_canvas_image(canvas_result) -> np.ndarray | None:
    """Pull the drawn RGBA canvas into a 2D uint8 grayscale array.
    Returns None if the canvas is empty (no strokes).

    Canvas renders a solid background (alpha=255 everywhere), so the
    alpha channel can't detect emptiness. We grayscale the RGB instead:
    strokes darken pixels, so a uniform-bright image means empty.
    """
    if canvas_result is None or canvas_result.image_data is None:
        return None
    rgba = canvas_result.image_data.astype(np.uint8)
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    if gray.min() > 240:
        return None
    # Invert so strokes are bright on dark background, matching the
    # MNIST-digit convention Member 1's thresholds are tuned for.
    return 255 - gray


# ─────────────────────────────────────────────
# Cached pipeline stages
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_member1(image: np.ndarray, low: int, high: int, stride: int):
    processed = preprocess_image(image)
    blurred = cv2.GaussianBlur(processed, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    coords = np.column_stack(np.where(edges > 0))
    coords = np.unique(coords, axis=0)
    if stride > 1:
        coords = coords[::stride]
    return processed, edges, coords


@st.cache_data(show_spinner=False)
def run_member2(coords: np.ndarray):
    clean, basis, rank, rref, sv = run_member2_pipeline(coords)
    return clean, basis, rank, sv


@st.cache_data(show_spinner=False)
def run_member3(clean: np.ndarray, basis: np.ndarray):
    return run_member3_pipeline(clean, basis)


@st.cache_data(show_spinner=False)
def run_member4(coords_2d: np.ndarray, depth: float):
    points_3d, edges = extrude_points(coords_2d, depth=depth)
    return points_3d, edges


@st.cache_data(show_spinner=False)
def run_member4_mesh(coords_2d: np.ndarray, depth: float, alpha: float):
    return extrude_mesh(coords_2d, depth=depth, alpha=alpha)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Sketch to 3D", layout="wide")
st.title("Sketch to 3D")
st.caption(
    "A linear-algebra pipeline: image -> edges -> matrix structure "
    "-> projection/eigen -> 3D extrusion."
)

with st.sidebar:
    st.header("Input")
    source = st.radio("Image source", ["MNIST digit", "Upload image", "Draw"])
    raw_image = None

    if source == "MNIST digit":
        digit_idx = st.slider("Sample index", 0, 1796, 0)
        raw_image = load_digit_sample(digit_idx)

    elif source == "Upload image":
        upload = st.file_uploader(
            "Upload a grayscale-friendly image", type=["png", "jpg", "jpeg", "bmp"]
        )
        if upload is None:
            st.info("Upload an image to continue.")
            st.stop()
        raw_image = load_uploaded_image(upload)

    else:  # Draw
        st.caption("Draw with the mouse — the alpha channel is the shape mask.")
        stroke_width = st.slider("Brush size", 1, 30, 6)
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#ffffff",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="sketch_canvas",
        )
        raw_image = load_canvas_image(canvas)
        if raw_image is None:
            st.info("Draw something on the canvas to continue.")
            st.stop()

    st.header("Edge detection")
    canny_low = st.slider("Canny low threshold", 0, 255, 30)
    canny_high = st.slider("Canny high threshold", 0, 255, 100)
    stride = st.slider("Coord stride (downsampling)", 1, 8, 2)

    st.header("3D extrusion")
    depth = st.slider("Depth (z)", 1.0, 100.0, 20.0)
    render_mode = st.radio("Render mode", ["Solid slab", "Point cloud", "Both"])
    alpha = st.slider(
        "Mesh alpha (max triangle edge)",
        1.0, 60.0, 12.0,
        help="Drops skinny Delaunay triangles. Raise for more fill, lower for tighter outline.",
    )
    show_walls = st.checkbox("Show walls (point cloud)", value=True)
    point_size = st.slider("Point size", 1, 6, 2)

# ── Member 1 ──
st.subheader("Member 1 - Image to sketch")
processed, edges_img, coords = run_member1(raw_image, canny_low, canny_high, stride)

c1, c2, c3 = st.columns(3)
c1.image(processed, caption="Preprocessed", width="stretch", clamp=True)
c2.image(edges_img, caption="Canny edges", width="stretch", clamp=True)
with c3:
    st.write(f"Edge points extracted: **{len(coords)}**")
    if len(coords) == 0:
        st.error("No edges found - try lowering the Canny thresholds.")
        st.stop()
    st.write("Sample coords (row, col):")
    st.dataframe(coords[:8])

# ── Member 2 ──
st.subheader("Member 2 - Matrix simplification and structure")
clean, basis, rank, sv = run_member2(coords)
m2a, m2b = st.columns([1, 2])
with m2a:
    st.metric("Unique points", clean.shape[0])
    st.metric("Rank", int(rank))
    st.write("Singular values:")
    st.write(np.round(sv, 4))
with m2b:
    st.write("Basis vectors (rows of V^T from SVD):")
    st.dataframe(np.round(basis, 4))

# ── Member 3 ──
st.subheader("Member 3 - Projection and eigen")
m3 = run_member3(clean, basis)
projected = m3["projected_points"]
eigvecs = m3["principal_directions"]
eigvals = m3["importance"]

m3a, m3b = st.columns([1, 2])
with m3a:
    st.write("Eigenvalues (importance):")
    st.write(np.round(eigvals, 4))
    total = float(eigvals.sum()) or 1.0
    st.write("Variance explained:")
    st.write(np.round(eigvals / total, 4))
with m3b:
    st.write("Principal directions (columns are eigenvectors):")
    st.dataframe(np.round(eigvecs, 4))

# ── Member 4 ──
st.subheader("Member 4 - 3D reconstruction")
points_3d, edges_idx = run_member4(clean, depth)
vertices, faces = run_member4_mesh(clean, depth, alpha)

if render_mode == "Point cloud":
    fig = build_figure(points_3d, edges_idx, show_walls=show_walls, point_size=point_size)
    st.plotly_chart(fig, use_container_width=True)
elif render_mode == "Solid slab":
    fig = build_mesh_figure(vertices, faces)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} triangles.")
else:  # Both
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            build_mesh_figure(vertices, faces),
            use_container_width=True,
        )
        st.caption(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} triangles.")
    with col_b:
        st.plotly_chart(
            build_figure(points_3d, edges_idx, show_walls=show_walls, point_size=point_size),
            use_container_width=True,
        )
    fig = build_mesh_figure(vertices, faces)

with st.expander("Download artefacts"):
    def _npy_bytes(arr):
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    st.download_button("coords.npy", _npy_bytes(coords), "coords.npy")
    st.download_button("clean_matrix.npy", _npy_bytes(clean), "clean_matrix.npy")
    st.download_button("basis_vectors.npy", _npy_bytes(basis), "basis_vectors.npy")
    st.download_button("projected_points.npy", _npy_bytes(projected), "projected_points.npy")
    st.download_button("points_3d.npy", _npy_bytes(points_3d), "points_3d.npy")
    st.download_button(
        "model_3d.html",
        fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
        "model_3d.html",
        mime="text/html",
    )
