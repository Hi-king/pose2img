import cv2
import cv2 as cv


class OpenPose(object):
    """
    This implementation is based on https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """

    def __init__(self, proto2d, model2d, thr=0.1, trained_dataset='COCO', reshape=(368, 368)):
        self.reshape = reshape
        self.trained_dataset = trained_dataset
        self.thr = thr
        self.net = cv.dnn.readNetFromCaffe(proto2d, model2d)
        self.body_parts, self.pose_pairs = self.parts(self.trained_dataset)

    def predict(self, frame):
        in_width = self.reshape[0]
        in_height = self.reshape[1]

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height),
                                   (0, 0, 0), swapRB=False)
        self.net.setInput(inp)
        out = self.net.forward()

        points = []
        for i in range(len(self.body_parts)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frame_width * point[0]) / out.shape[3]
            y = (frame_height * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((x, y) if conf > self.thr else None)
        return points

    def show(self, float_points, frame):
        points = []
        for point in float_points:
            if point is None:
                points.append(None)
            else:
                points.append(tuple([int(value) for value in point]))
        for pair in self.pose_pairs:
            part_from = pair[0]
            part_to = pair[1]
            assert (part_from in self.body_parts)
            assert (part_to in self.body_parts)

            id_from = self.body_parts[part_from]
            id_to = self.body_parts[part_to]

            if points[id_from] and points[id_to]:
                cv.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
                cv.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        return frame

    @staticmethod
    def parts(trained_dataset):
        if trained_dataset == 'COCO':
            body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                          "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                          "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                          "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

            pose_pairs = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                          ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                          ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                          ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                          ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
        else:
            assert (trained_dataset == 'MPI')
            body_parts = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                          "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                          "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                          "Background": 15}

            pose_pairs = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                          ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                          ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                          ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
        return body_parts, pose_pairs
