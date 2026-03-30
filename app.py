import os
from shutil import copyfile
from functools import wraps, update_wrapper
from datetime import datetime

from flask import Flask, render_template, request, make_response
import image_processing

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(APP_ROOT, "static", "img")


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = (
            'no-store, no-cache, must-revalidate, '
            'post-check=0, pre-check=0, max-age=0'
        )
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


def ensure_img_dir():
    os.makedirs(IMG_DIR, exist_ok=True)


def uploaded_response():
    return render_template("uploaded.html", file_path="img/img_now.jpg")


def run_processing(fn, *args):
    """Wrap an image_processing call and return the result page, or an error page on failure."""
    try:
        if args:
            fn(*args)
        else:
            fn()
        return uploaded_response()
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


# ─────────────────────────────────────────────
#  ROUTES — PAGES
# ─────────────────────────────────────────────

@app.route("/")
@app.route("/index")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template("about.html")


# ─────────────────────────────────────────────
#  ROUTES — UPLOAD
# ─────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
@nocache
def upload():
    ensure_img_dir()
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        return render_template("home.html", file_path="img/image_here.jpg",
                               error="No file selected."), 400
    files[0].save(os.path.join(IMG_DIR, "img_now.jpg"))
    copyfile(
        os.path.join(IMG_DIR, "img_now.jpg"),
        os.path.join(IMG_DIR, "img_normal.jpg")
    )
    return uploaded_response()


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile(
        os.path.join(IMG_DIR, "img_normal.jpg"),
        os.path.join(IMG_DIR, "img_now.jpg")
    )
    return uploaded_response()


# ─────────────────────────────────────────────
#  ROUTES — BASIC OPERATIONS
# ─────────────────────────────────────────────

@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    return run_processing(image_processing.grayscale)


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    return run_processing(image_processing.zoomin)


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    return run_processing(image_processing.zoomout)


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    return run_processing(image_processing.move_left)


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    return run_processing(image_processing.move_right)


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    return run_processing(image_processing.move_up)


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    return run_processing(image_processing.move_down)


# ─────────────────────────────────────────────
#  ROUTES — BRIGHTNESS
# ─────────────────────────────────────────────

@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    return run_processing(image_processing.brightness_addition)


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    return run_processing(image_processing.brightness_substraction)


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    return run_processing(image_processing.brightness_multiplication)


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    return run_processing(image_processing.brightness_division)


# ─────────────────────────────────────────────
#  ROUTES — LOW-LIGHT ENHANCEMENT  (new)
# ─────────────────────────────────────────────

@app.route("/gamma_correction", methods=["POST"])
@nocache
def gamma_correction():
    gamma = float(request.form.get("gamma", 0.5))
    gamma = max(0.1, min(gamma, 2.0))   # clamp to safe range
    return run_processing(image_processing.gamma_correction, gamma)


@app.route("/clahe", methods=["POST"])
@nocache
def clahe():
    return run_processing(image_processing.clahe_enhancement)


@app.route("/retinex", methods=["POST"])
@nocache
def retinex():
    return run_processing(image_processing.retinex_enhancement)


@app.route("/denoise", methods=["POST"])
@nocache
def denoise():
    return run_processing(image_processing.denoise)


# ─────────────────────────────────────────────
#  ROUTES — FILTERS
# ─────────────────────────────────────────────

@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    return run_processing(image_processing.edge_detection)


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    return run_processing(image_processing.blur)


@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    return run_processing(image_processing.sharpening)


# ─────────────────────────────────────────────
#  ROUTES — HISTOGRAM / THRESHOLD
# ─────────────────────────────────────────────

@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    return run_processing(image_processing.histogram_equalizer)


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    try:
        image_processing.histogram_rgb()
        if image_processing.is_grey_scale():
            return render_template("histogram.html",
                                   file_paths=["img/grey_histogram.jpg"])
        else:
            return render_template("histogram.html",
                                   file_paths=["img/red_histogram.jpg",
                                               "img/green_histogram.jpg",
                                               "img/blue_histogram.jpg"])
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    try:
        lower = int(request.form.get("lower_thres", 0))
        upper = int(request.form.get("upper_thres", 255))
        return run_processing(image_processing.threshold, lower, upper)
    except ValueError:
        return render_template("error.html",
                               error="Threshold values must be integers."), 400


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ensure_img_dir()
    app.run(port=5001, debug=False)
