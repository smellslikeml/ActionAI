

if __name__ == '__main__':
    import cv2
    import pickle
    import argparse
    import importlib
    from transformer import PoseExtractor

    parser = argparse.ArgumentParser(description='Run inference on webcam video')
    parser.add_argument('--config', type=str, default='conf',
                        help="name of config .py file inside config/ directory, default: 'conf'")
    args = parser.parse_args()
    config = importlib.import_module('config.' + args.config)

    model = pickle.load(open(config.classifier_model, 'rb'))

    extractor = PoseExtractor()
    cap = cv2.VideoCapture(config.stream)
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == True:
            sample = extractor.transform([image])
            prediction = model.predict(sample.reshape(1, -1))
            print(prediction[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()



