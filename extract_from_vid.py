import cv2

def extract_video(video, interval):
    '''
    For extracting the images from the video.

    Video: the path of the video
    Interval: how many seconds for the next image capturing
    Img_name: the name of the saved image, without the file extension
    '''

    vid = cv2.VideoCapture(video)
    fps = round(vid.get(cv2.CAP_PROP_FPS))
    print('FPS: ', fps)
    hop = fps*interval
    curr_frame = 0
    count = 0
    while True:
        ret, frame = vid.read()
        if not ret: 
            break
        if curr_frame % hop == 0:
            name = 'Image_' + str(count) + '.png'
            cv2.imwrite(name, frame)
            count += 1
            print(curr_frame)

        curr_frame += 1
        
    vid.release()

extract_video('concrete_numbers00_preview.mp4', 1)
