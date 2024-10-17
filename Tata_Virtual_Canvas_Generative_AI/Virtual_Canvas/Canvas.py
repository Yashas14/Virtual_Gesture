import mediapipe as mp
import cv2
import numpy as np
import time

# Constants
ml = 150
color_panel_width = 150
color_panel_height = 50
curr_tool = None
curr_color = None
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0
tool_selected = False

# Define panel size for tools
max_x, max_y = 250 + ml, 50

# Get tools function
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    elif x < 250 + ml:
        return "erase"
    else:
        return "none"

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    else:
        return False

# Initialize MediaPipe hands
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

# Resize tools to fit the region
tools_height, tools_width = tools.shape[:2]
tools_resized = cv2.resize(tools, (max_x - ml, 50))

mask = np.ones((480, 640, 3)) * 255
mask = mask.astype('uint8')

# Create color selection panel
color_panel = np.zeros((color_panel_height, color_panel_width, 3), dtype="uint8")
colors = [(0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Yellow, Green, Red, Blue
for i, color in enumerate(colors):
    cv2.rectangle(color_panel, (i * (color_panel_width // len(colors)), 0),
                  ((i + 1) * (color_panel_width // len(colors)), color_panel_height), color, -1)
    cv2.putText(color_panel, str(i + 1), (i * (color_panel_width // len(colors)) + 10, color_panel_height // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frm = cap.read()
    if not ret:
        print("Error: Frame capture failed, exiting.")
        break

    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            # Color selection logic
            if y < color_panel_height:
                if x < color_panel_width:
                    index = x // (color_panel_width // len(colors))
                    if index < len(colors):
                        curr_color = colors[index]
                        print("Selected color:", curr_color)

            # Tool selection logic
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Current tool set to:", curr_tool)
                    time_init = True
                    rad = 40
                    tool_selected = True

            else:
                time_init = True
                rad = 40

            # Drawing logic
            if curr_tool and curr_color and tool_selected:
                if curr_tool == "draw":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if prevx == 0 and prevy == 0:
                            prevx, prevy = x, y  # Initialize previous coordinates for first draw
                        cv2.line(mask, (prevx, prevy), (x, y), curr_color, thick)
                        prevx, prevy = x, y
                    else:
                        prevx = x
                        prevy = y

                elif curr_tool == "line":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True
                        cv2.line(frm, (xii, yii), (x, y), curr_color, thick)
                    else:
                        if var_inits:
                            cv2.line(mask, (xii, yii), (x, y), curr_color, thick)
                            var_inits = False

                elif curr_tool == "rectangle":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True
                        cv2.rectangle(frm, (xii, yii), (x, y), curr_color, thick)
                    else:
                        if var_inits:
                            cv2.rectangle(mask, (xii, yii), (x, y), curr_color, thick)
                            var_inits = False

                elif curr_tool == "circle":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True
                        # Use Euclidean distance to calculate radius
                        radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                        cv2.circle(frm, (xii, yii), radius, curr_color, thick)
                    else:
                        if var_inits:
                            radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                            cv2.circle(mask, (xii, yii), radius, curr_color, thick)
                            var_inits = False

                elif curr_tool == "erase":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        cv2.circle(mask, (x, y), 30, (255, 255, 255), -1)

    # Apply the color panel to the top-left corner of the frame
    color_panel_resized = cv2.resize(color_panel, (color_panel_width, color_panel_height))
    frm[:color_panel_height, :color_panel_width] = color_panel_resized

    # Ensure the tools image fits the frame region
    tools_resized = cv2.resize(tools, (max_x - ml, 50))  # Resize tools to the correct region size
    frm[:max_y, ml:max_x] = cv2.addWeighted(tools_resized, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

    # Apply the mask to the frame
    op = cv2.bitwise_and(frm, mask, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    frm[:, :, 0] = op[:, :, 0]
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    cv2.putText(frm, f"Tool: {curr_tool}" if curr_tool else "Select Tool", (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frm, f"Color: {curr_color}" if curr_color else "Select Color", (270 + ml, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("paint app", frm)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
