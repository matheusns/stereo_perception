# Pedro Video Recorder
import cv2
import numpy as np

if __name__ == "__main__":
    
    rgb = cv2.imread("sample_rgb.png")
    mono = cv2.imread("sample_mono.png")

    rgb_bounded = cv2.resize(rgb, (640, 360))
    mono_bounded = cv2.resize(mono, (640, 360))
    

    temp = np.vstack([np.hstack([mono_bounded,rgb_bounded]), np.hstack([mono_bounded,rgb_bounded])])
    cv2.namedWindow('Pedrisco', cv2.WINDOW_NORMAL)
    cv2.imshow('Pedrisco', temp)
    key = cv2.waitKey(0)

    cap = cv2.VideoCapture(0)

    # Video output setup
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Converts to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # This operation is needed because "vstack" ask for data from the same type in this case BGR
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


        # You can feel free to add as many images as you want, however, pay 
        # attention to the encoding (bgr,mono) and the size that must be the
        # same for all of them
        temp = np.vstack([np.hstack([frame,gray]), np.hstack([frame,gray])])
        # The video size must be equal to the set before
        temp_bounded = cv2.resize(temp, (1280, 720))
        out.write(temp_bounded)

        cv2.imshow('frame',temp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release all captures
    print "Releasing captures ..."
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Good presentation, see you soon!!
# May the force be with you!!!