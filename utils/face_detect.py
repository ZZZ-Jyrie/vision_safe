import cv2
import mediapipe as mp
import numpy as np


class FaseDetector:

    def __init__(self):
        ############## PARAMETERS #######################################################

        # Set these values to show/hide certain vectors of the estimation
        self.draw_gaze = True
        self.draw_full_axis = True
        self.draw_headpose = False

        # Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
        self.x_score_multiplier = 10
        self.y_score_multiplier = 10

        # Threshold of how close scores should be to average between frames
        self.threshold = .3

        #################################################################################

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                               refine_landmarks=True,
                                               max_num_faces=2,
                                               min_detection_confidence=0.5)

        face_3d = np.array([
            [0.0, 0.0, 0.0],  # Nose tip
            [0.0, -330.0, -65.0],  # Chin
            [-225.0, 170.0, -135.0],  # Left eye left corner
            [225.0, 170.0, -135.0],  # Right eye right corner
            [-150.0, -150.0, -125.0],  # Left Mouth corner
            [150.0, -150.0, -125.0]  # Right mouth corner
        ], dtype=np.float64)

        # Reposition left eye corner to be the origin
        self.leye_3d = np.array(face_3d)
        self.leye_3d[:, 0] += 225
        self.leye_3d[:, 1] -= 175
        self.leye_3d[:, 2] += 135

        # Reposition right eye corner to be the origin
        self.reye_3d = np.array(face_3d)
        self.reye_3d[:, 0] -= 225
        self.reye_3d[:, 1] -= 175
        self.reye_3d[:, 2] += 135

        # Gaze scores from the previous frame
        self.last_lx, self.last_rx = 0, 0
        self.last_ly, self.last_ry = 0, 0

    def detect_face(self, img):
        # Flip + convert img from BGR to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        img.flags.writeable = False

        # Get the result
        results = self.face_mesh.process(img)
        img.flags.writeable = True

        # Convert the color space from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        (img_h, img_w, img_c) = img.shape
        face_2d = []

        if not results.multi_face_landmarks:
            return img

        for face_landmarks in results.multi_face_landmarks:
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                # Convert landmark x and y to pixel coordinates
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Add the 2D coordinates to an array
                face_2d.append((x, y))

            # Get relevant landmarks for headpose estimation
            face_2d_head = np.array([
                face_2d[1],  # Nose
                face_2d[199],  # Chin
                face_2d[33],  # Left eye left corner
                face_2d[263],  # Right eye right corner
                face_2d[61],  # Left mouth corner
                face_2d[291]  # Right mouth corner
            ], dtype=np.float64)

            face_2d = np.asarray(face_2d)

            # Calculate left x gaze score
            if (face_2d[243, 0] - face_2d[130, 0]) != 0:
                lx_score = (face_2d[468, 0] - face_2d[130, 0]) / (face_2d[243, 0] - face_2d[130, 0])
                if abs(lx_score - self.last_lx) < self.threshold:
                    lx_score = (lx_score + self.last_lx) / 2
                self.last_lx = lx_score

            # Calculate left y gaze score
            if (face_2d[23, 1] - face_2d[27, 1]) != 0:
                ly_score = (face_2d[468, 1] - face_2d[27, 1]) / (face_2d[23, 1] - face_2d[27, 1])
                if abs(ly_score - self.last_ly) < self.threshold:
                    ly_score = (ly_score + self.last_ly) / 2
                self.last_ly = ly_score

            # Calculate right x gaze score
            if (face_2d[359, 0] - face_2d[463, 0]) != 0:
                rx_score = (face_2d[473, 0] - face_2d[463, 0]) / (face_2d[359, 0] - face_2d[463, 0])
                if abs(rx_score - self.last_rx) < self.threshold:
                    rx_score = (rx_score + self.last_rx) / 2
                self.last_rx = rx_score

            # Calculate right y gaze score
            if (face_2d[253, 1] - face_2d[257, 1]) != 0:
                ry_score = (face_2d[473, 1] - face_2d[257, 1]) / (face_2d[253, 1] - face_2d[257, 1])
                if abs(ry_score - self.last_ry) < self.threshold:
                    ry_score = (ry_score + self.last_ry) / 2
                self.last_ry = ry_score

            # The camera matrix
            self.focal_length = 1 * img_w
            cam_matrix = np.array([[self.focal_length, 0, img_h / 2],
                                   [0, self.focal_length, img_w / 2],
                                   [0, 0, 1]])

            # Distortion coefficients
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            _, l_rvec, l_tvec = cv2.solvePnP(self.leye_3d, face_2d_head, cam_matrix, dist_coeffs,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
            _, r_rvec, r_tvec = cv2.solvePnP(self.reye_3d, face_2d_head, cam_matrix, dist_coeffs,
                                             flags=cv2.SOLVEPNP_ITERATIVE)

            # Get rotational matrix from rotational vector
            l_rmat, _ = cv2.Rodrigues(l_rvec)
            r_rmat, _ = cv2.Rodrigues(r_rvec)

            # Adjust headpose vector with gaze score
            l_gaze_rvec = np.array(l_rvec)
            l_gaze_rvec[2][0] -= (lx_score - .5) * self.x_score_multiplier
            l_gaze_rvec[0][0] += (ly_score - .5) * self.y_score_multiplier

            r_gaze_rvec = np.array(r_rvec)
            r_gaze_rvec[2][0] -= (rx_score - .5) * self.x_score_multiplier
            r_gaze_rvec[0][0] += (ry_score - .5) * self.y_score_multiplier

            # --- Projection ---

            # Get left eye corner as integer
            l_corner = face_2d_head[2].astype(np.int32)

            # Project axis of rotation for left eye
            axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
            l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
            l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)

            # Draw axis of rotation for left eye
            if self.draw_headpose:
                if self.draw_full_axis:
                    cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (200, 200, 0), 3)
                    cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (0, 200, 0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0, 200, 200), 3)

            if self.draw_gaze:
                if self.draw_full_axis:
                    cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (255, 0, 0), 3)
                    cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (0, 255, 0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0, 0, 255), 3)
                left_eye_gaze_vector = l_gaze_axis[2].ravel().tolist()
                print("Left Eye Gaze Vector:", left_eye_gaze_vector)
                cv2.putText(img, f"L: {left_eye_gaze_vector}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Get right eye corner as integer
            r_corner = face_2d_head[3].astype(np.int32)

            # Get right eye corner as integer
            r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
            r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

            # Draw axis of rotation for right eye
            if self.draw_headpose:
                if self.draw_full_axis:
                    cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200, 200, 0), 3)
                    cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0, 200, 0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0, 200, 200), 3)

            if self.draw_gaze:
                if self.draw_full_axis:
                    cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255, 0, 0), 3)
                    cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0, 255, 0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0, 0, 255), 3)
                right_eye_gaze_vector = r_gaze_axis[2].ravel().tolist()
                print("Right Eye Gaze Vector:", right_eye_gaze_vector)
                cv2.putText(img, f"R: {right_eye_gaze_vector}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img
