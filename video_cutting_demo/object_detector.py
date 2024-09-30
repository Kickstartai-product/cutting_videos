import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
from time import time

class ObjectDetector:
    def __init__(self, output_frames_dir="output_frames", confidence_threshold=0.5, verbose=False, save_frames=False):
        self.output_frames_dir = output_frames_dir
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.RELEVANT_CLASSES = {
            1: "person",
            25: "backpack",
            27: "handbag",
            29: "suitcase"
        }
        self.save_frames = save_frames
        
        os.makedirs(self.output_frames_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.verbose = verbose

    def predict(self, frame, frame_count):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            start_time = time()
            predictions = self.model(image_tensor)
            if self.verbose:
                print(f'Time taken: {time() - start_time:.2f} seconds')
        
        detected_objects = []
        
        for prediction in predictions:
            boxes = prediction["boxes"]
            labels = prediction["labels"]
            scores = prediction["scores"]
            
            relevant_indices = [
                i
                for i, (label, score) in enumerate(zip(labels, scores))
                if label.item() in self.RELEVANT_CLASSES
                and score.item() > self.CONFIDENCE_THRESHOLD
            ]
            
            relevant_boxes = boxes[relevant_indices]
            relevant_labels = [
                self.RELEVANT_CLASSES[labels[i].item()] for i in relevant_indices
            ]
            relevant_scores = scores[relevant_indices]
            
            if self.save_frames:
                self.save_frame(
                    frame, frame_count, relevant_boxes, relevant_labels, relevant_scores
                )
            
            for box, label, score in zip(relevant_boxes, relevant_labels, relevant_scores):
                x1, y1, x2, y2 = box.int().tolist()
                detected_objects.append({
                    "name": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": score.item()
                })
        
        return detected_objects

    def save_frame(self, frame, frame_count, boxes=None, labels=None, scores=None):
        if boxes is not None and labels is not None and scores is not None:
            frame = self.draw_bounding_boxes(frame.copy(), boxes, labels, scores)
        
        output_path = os.path.join(
            self.output_frames_dir, f"frame_{frame_count:06d}.jpg"
        )
        cv2.imwrite(output_path, frame)

    def draw_bounding_boxes(self, image, boxes, labels, scores):
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(
                image,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        return image