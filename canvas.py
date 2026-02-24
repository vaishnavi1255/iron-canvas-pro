import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading

# Try to import Windows sound library
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ------------------------------
# Configuration settings
# ------------------------------
class Config:
    WIDTH, HEIGHT = 1280, 720
    PINCH_THRESHOLD = 40
    SMOOTHING = 0.6
    BRUSH_SIZE = 8
    HUD_COLOR = (255, 255, 0)

    ARC_CENTER = (640, 0)
    ARC_RADIUS = 150
    ARC_THICKNESS = 60


# ------------------------------
# Sound engine
# ------------------------------
class AudioFeedback:
    def __init__(self):
        self.active = False
        self.velocity = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()

    def set_drawing(self, is_drawing, velocity):
        self.active = is_drawing
        self.velocity = velocity

    def _loop(self):
        while not self.stop_event.is_set():
            if AUDIO_AVAILABLE and self.active:
                try:
                    freq = int(200 + (self.velocity * 5))
                    freq = max(100, min(freq, 800))
                    winsound.Beep(freq, 40)
                except:
                    pass
            else:
                time.sleep(0.05)


# ------------------------------
#  Setup gesture recognition engine
# ------------------------------
class GestureTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def process(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            h, w, _ = img.shape
            points = []
            for lm in landmarks:
                points.append((int(lm.x * w), int(lm.y * h)))
            return points
        return None

    def draw_hud(self, img, points, pinch_dist):
        if not points:
            return img

        overlay = img.copy()

        for pt in points:
            cv2.circle(overlay, pt, 4, (0, 255, 255), -1)

        idx_x, idx_y = points[8]
        cv2.circle(overlay, (idx_x, idx_y), 10, (255, 255, 255), 1)

        if pinch_dist < Config.PINCH_THRESHOLD:
            cv2.putText(
                overlay,
                "DRAW",
                (idx_x + 20, idx_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)


# ------------------------------
# Render radial color selection interface
# ------------------------------
class RadialColorUI:
    def __init__(self):
        self.colors = [
            ((0, 0, 255), "RED"),
            ((0, 165, 255), "ORANGE"),
            ((0, 255, 255), "YELLOW"),
            ((0, 255, 0), "GREEN"),
            ((255, 255, 0), "CYAN"),
            ((255, 0, 255), "PURPLE"),
            ((255, 255, 255), "WHITE"),
            ((0, 0, 0), "CLEAR"),
        ]
        self.selected_index = 4

    def draw(self, img, hover_pt):
        num_colors = len(self.colors)
        sector_angle = 180 / num_colors

        cx, cy = Config.ARC_CENTER
        radius = Config.ARC_RADIUS

        hover_index = -1

        if hover_pt:
            hx, hy = hover_pt
            dist = math.hypot(hx - cx, hy - cy)

            if radius < dist < radius + Config.ARC_THICKNESS:
                dx, dy = hx - cx, hy - cy
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0:
                    angle += 360
                if 0 <= angle <= 180:
                    hover_index = int(angle / sector_angle)

        for i in range(num_colors):
            start_ang = i * sector_angle
            end_ang = (i + 1) * sector_angle
            color, name = self.colors[i]

            thickness = Config.ARC_THICKNESS
            if i == self.selected_index:
                cv2.ellipse(
                    img,
                    (cx, cy),
                    (radius + 15, radius + 15),
                    0,
                    start_ang,
                    end_ang,
                    (255, 255, 255),
                    -1,
                )

            if i == hover_index:
                cv2.putText(
                    img,
                    name,
                    (cx - 40, cy + radius + 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

            cv2.ellipse(
                img,
                (cx, cy),
                (radius + (thickness // 2), radius + (thickness // 2)),
                0,
                start_ang,
                end_ang,
                color,
                thickness,
            )

        return hover_index


# ------------------------------
# Main Application
# ------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, Config.WIDTH)
    cap.set(4, Config.HEIGHT)

    hand_sys = HandSystem()
    palette = ArcPalette()
    sound = SoundEngine()

    canvas = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)

    smooth_x, smooth_y = 0, 0
    current_color = (255, 255, 0)

    shape_mode = False
    shape_type = "RECTANGLE"
    start_point = None

    print("IRON CANVAS PRO ACTIVATED")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        points = hand_sys.process(img)

        is_drawing = False
        velocity = 0

        if points:
            idx_tip = points[8]
            thm_tip = points[4]
            mid_tip = points[12]
            ring_tip = points[16]

            cx, cy = idx_tip

            if smooth_x == 0:
                smooth_x, smooth_y = cx, cy

            smooth_x = int(smooth_x * (1 - Config.SMOOTHING) + cx * Config.SMOOTHING)
            smooth_y = int(smooth_y * (1 - Config.SMOOTHING) + cy * Config.SMOOTHING)

            dist = math.hypot(idx_tip[0] - thm_tip[0], idx_tip[1] - thm_tip[1])

            img = hand_sys.draw_hud(img, points, dist)
            hover_idx = palette.draw(img, (smooth_x, smooth_y))

            # Finger counting
            fingers_up = 0
            if idx_tip[1] < points[6][1]:
                fingers_up += 1
            if mid_tip[1] < points[10][1]:
                fingers_up += 1
            if ring_tip[1] < points[14][1]:
                fingers_up += 1

            # Brush size control
            if fingers_up == 1:
                Config.BRUSH_SIZE = min(50, Config.BRUSH_SIZE + 1)
            if fingers_up == 2:
                Config.BRUSH_SIZE = max(2, Config.BRUSH_SIZE - 1)

            # Toggle shape mode
            if fingers_up == 3:
                shape_mode = not shape_mode
                time.sleep(0.4)

            # Palette selection
            if hover_idx != -1 and dist < Config.PINCH_THRESHOLD:
                color, name = palette.colors[hover_idx]
                if name == "CLEAR":
                    canvas[:] = 0
                else:
                    palette.selected_index = hover_idx
                    current_color = color

            elif dist < Config.PINCH_THRESHOLD and smooth_y > 200:

                if not shape_mode:
                    is_drawing = True
                    velocity = math.hypot(smooth_x - cx, smooth_y - cy)

                    cv2.line(canvas, (smooth_x, smooth_y), (cx, cy),
                             current_color, Config.BRUSH_SIZE)

                else:
                    if start_point is None:
                        start_point = (cx, cy)

                    temp_canvas = canvas.copy()

                    if shape_type == "RECTANGLE":
                        cv2.rectangle(temp_canvas, start_point, (cx, cy),
                                      current_color, Config.BRUSH_SIZE)
                    else:
                        radius = int(math.hypot(cx - start_point[0],
                                                cy - start_point[1]))
                        cv2.circle(temp_canvas, start_point, radius,
                                   current_color, Config.BRUSH_SIZE)

                    canvas = temp_canvas
            else:
                start_point = None

            smooth_x, smooth_y = cx, cy
        else:
            palette.draw(img, None)

        sound.set_drawing(is_drawing, velocity)

        # Glow Effect
        canvas_small = cv2.resize(canvas, (0, 0), fx=0.2, fy=0.2)
        blur = cv2.GaussianBlur(canvas_small, (15, 15), 0)
        blur_up = cv2.resize(blur, (Config.WIDTH, Config.HEIGHT))
        final_canvas = cv2.addWeighted(canvas, 1.0, blur_up, 1.5, 0)

        gray = cv2.cvtColor(final_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        img = cv2.add(img_bg, final_canvas)

        # UI Text
        cv2.putText(img, f"Brush Size: {Config.BRUSH_SIZE}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        mode_text = "SHAPE MODE" if shape_mode else "DRAW MODE"
        cv2.putText(img, mode_text,
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow("Iron Canvas Pro", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"iron_canvas_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print("Image Saved:", filename)
        elif key == ord("x"):
            canvas[:] = 0
        elif key == ord("c"):
            shape_type = "CIRCLE" if shape_type == "RECTANGLE" else "RECTANGLE"

    sound.stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()