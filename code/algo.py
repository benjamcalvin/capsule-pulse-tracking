
import cv2 as cv
import numpy as np
import pandas as pd
import os
import re
import time

### SET THE PARAMETERS FOR THE RUN:

def get_params(rawfilename, directory=None, is_job=False):

    params = dict()

    params['videoname'] = re.split("\.", rawfilename)[0]

    if is_job:
        params['directory'] = '/storage/'
        params['output_directory'] = '/artifacts/'
    else:
        if directory is not None:
            params['directory'] = f"../../videos/{directory}/"
            params['output_directory'] = f"../../videos/{directory}/"
        else:
            params['directory'] = f"../../videos/{params['videoname']}/"
            params['output_directory'] = f"../../videos/{params['videoname']}/"

    return params



def resize(rawfilename, reduction = 2, directory=None, is_job=False):

    params = get_params(rawfilename, directory, is_job)

    if os.path.exists(f'{params["directory"]}{params["videoname"]}_resized.avi'):
        print("The file "+f'{params["directory"]}{params["videoname"]}_resized.avi'+" already exists. Please rename, remove, or delete to denoise again. Skipping...")
        return False

    vc = cv.VideoCapture(params['directory']+rawfilename)

    def resize_helper(img, fact=.5):
        return(cv.resize(img, (0,0), fx=fact, fy=fact))

    success, img = vc.read()
    frame = 0

    height, width, layers = img.shape
    fourcc = cv.VideoWriter_fourcc(*'FFV1')
    video_out = cv.VideoWriter(f'{params["directory"]}{params["videoname"]}_resized.avi', fourcc, 30, (width//reduction, height//reduction))

    while success:
        img=resize_helper(img, fact=1./reduction)
        video_out.write(img)
        success, img = vc.read()
        frame += 1
        
        if ((frame % 1000) == 0):
            print(f"Done with frame {frame}")
        
    cv.destroyAllWindows()
    video_out.release()

    return True

def denoise(rawfilename, h=10, templateWindowSize=7, searchWindowPixel=35, is_job=False, minframe=None, maxframe=None, test=None, directory=None):

    params = get_params(rawfilename, directory, is_job)

    if minframe is None:
        minframe = 0
    if maxframe is None:
        maxframe = 100000

    vc = cv.VideoCapture(f'{params["directory"]}{params["videoname"]}_resized.avi')

    if test is not None:
        params["videoname"] += test

    # Break if video already exists... tells you to remove:
    if os.path.exists(f'{params["directory"]}{params["videoname"]}_resized_denoised_{h}_{templateWindowSize}_{searchWindowPixel}.avi'):
        print("The file "+f'{params["directory"]}{params["videoname"]}_resized_denoised_{h}_{templateWindowSize}_{searchWindowPixel}.avi'+" already exists. Please rename, remove, or delete to denoise again. Skipping...")
        return False

    def denoise(imgs):
        return(cv.fastNlMeansDenoisingMulti(imgs, 2, 5, None, h, templateWindowSize, searchWindowPixel))

    success, img1 = vc.read()
    success, img2 = vc.read()
    success, img3 = vc.read()
    success, img4 = vc.read()
    success, img5 = vc.read()
    frame = 0

    height, width, layers = img1.shape
    fourcc = cv.VideoWriter_fourcc(*'FFV1')

    video_out = cv.VideoWriter(f'{params["directory"]}{params["videoname"]}_resized_denoised_{h}_{templateWindowSize}_{searchWindowPixel}.avi', fourcc, 30, (width, height))

    while (success):
        if (frame >= minframe) and (frame <= maxframe): 
            img = denoise([img1, img2, img3, img4, img5])
            video_out.write(img)
        img1=img2.copy()
        img2=img3.copy()
        img3=img4.copy()
        img4=img5.copy()
        success, img5 = vc.read()
        frame += 1
        
        if ((frame % 100) == 0):
            print(f"Done with frame {frame}")

    cv.destroyAllWindows()
    video_out.release()

    return True

def extract_motion(rawfilename, is_job=False, th=[7, 9, 11],
        history=15, h=10, templateWindowSize=7, searchWindowPixel=35,
        write_video=True, minframe=None, maxframe=None, test=None,
        directory=None, bw=False, remove_flicker=False):

    # Start time:
    start = time.time()

    # Get parameters
    params = get_params(rawfilename, directory, is_job)

    if test is not None:
        params["videoname"] += test

    if minframe is None:
        minframe = 0
    if maxframe is None:
        maxframe = 100000

    # Open video
    mv = cv.VideoCapture(f'{params["directory"]}{params["videoname"]}_resized_denoised_{h}_{templateWindowSize}_{searchWindowPixel}.avi')
    success, frame = mv.read()
    if not(success):
        raise Exception("Could not read video: "+f'{params["directory"]}{params["videoname"]}_resized_denoised_{h}_{templateWindowSize}_{searchWindowPixel}.avi'+" check to make sure it exists.")
    height, width, layers = frame.shape
    f = 1

    # Data constructs to save the measurements
    sum_of_diffs = dict()
    num_of_diffs = dict()

    # Initialize the constructs with empty lists for each threshold, so we can eventually add a value for each frame for each threshold:
    for t in th:
        sum_of_diffs[t] = list()
        num_of_diffs[t] = list()

    if not(os.path.isdir(f'{params["output_directory"]}motion_past_{history}_frames')):
        os.makedirs(f'{params["output_directory"]}motion_past_{history}_frames')

    if bw:
        csv_name = f'{params["output_directory"]}motion_past_{history}_frames/{params["videoname"]}_{"and".join([f"{t}" for t in th])}_{h}_{templateWindowSize}_{searchWindowPixel}_bw.csv'
    else:
        csv_name = f'{params["output_directory"]}motion_past_{history}_frames/{params["videoname"]}_{"and".join([f"{t}" for t in th])}_{h}_{templateWindowSize}_{searchWindowPixel}.csv'

    #  Initialize necessary directories if writing video
    if write_video:
        if bw:
            video_file = f'{params["output_directory"]}motion_past_{history}_frames/{params["videoname"]}_{"and".join([f"{t}" for t in th])}_{h}_{templateWindowSize}_{searchWindowPixel}_bw.avi'
        else:
            video_file = f'{params["output_directory"]}motion_past_{history}_frames/{params["videoname"]}_{"and".join([f"{t}" for t in th])}_{h}_{templateWindowSize}_{searchWindowPixel}.avi'

        if os.path.exists(video_file):
            raise Exception("It appears you've already run this configuration: history ({history}, th ({th}) for this configuration, not writing video. Delete, move, or rename the existing file to run this configuration. Check here "+ f'{params["output_directory"]}motion_past_{history}_frames/' + 'or you can simply set write_video to False')
        fourcc = cv.VideoWriter_fourcc(*'FFV1')
        video_out = cv.VideoWriter(video_file,  fourcc, 30, (width*(len(th)+1), height))

    else:
        if os.path.exists(csv_name):
            print("Data already exists...")
            return pd.read_csv(csv_name)


    # Initialize dictionary construct:
    prev_frames = dict()
    diffs = dict()

    while(success):
        if write_video:
            output_frame = frame.copy()
            cv.putText(output_frame, "Frame {}".format(f), (200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        if bw:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if ((f > history) and (f >= minframe) and (f <= maxframe)):
            # Take the difference between pixel values in the current frame and the previous {number value of history} frames
            for hi in range(history):
                diffs[hi] = cv.absdiff(prev_frames[h], frame)
            
            # Take the maximum amount each pixel has changed between now and any of the past 15 frames
            max_diff = np.maximum(diffs[0], diffs[1])
            for hi in range(2, history):
                max_diff = np.maximum(max_diff, diffs[hi])

            # Loop over each threshold value you want to test, and eliminate differences below the thresholds:
            for t in th:
                # Record the sum of differences above the threshold:
                if bw:
                    sum_of_diffs[t].append(max(max_diff[max_diff[:, :] > t].sum(), 1))
                    num_of_diffs[t].append(max(diffs[0][diffs[0] > t].shape[0], 1))
                else:
                    sum_of_diffs[t].append(max(max_diff[max_diff[:, :, :] > t].sum(), 1))
                    num_of_diffs[t].append(max(diffs[0][diffs[0] > t].shape[0], 1))

                # For video out:
                if write_video:
                    diff_th = max_diff.copy()
                    diff_th[max_diff > t] = 255
                    diff_th[max_diff <= t] = 0
                    if bw:
                        diff_th = cv.cvtColor(diff_th, cv.COLOR_GRAY2BGR)
                    output_frame = np.concatenate((output_frame, diff_th), axis=1)

            if write_video:
                video_out.write(output_frame)

        for hi in range(min(f, history), 1, -1):
            prev_frames[hi-1] = prev_frames[hi-2].copy()

        prev_frames[0] = frame.copy()
        success, frame = mv.read()

        f += 1

    cv.destroyAllWindows()
    if write_video:
        video_out.release()

    # Write out data:
    data = pd.DataFrame(list(range(len(sum_of_diffs[th[0]]))))
    data = data.rename(index=str, columns={0:"frame"})
    data["frame"] = data["frame"] + 15
    data.index = list(range(data.shape[0]))

    for t in th:
        data = data.merge(pd.DataFrame(sum_of_diffs[t]), left_index=True, right_index=True, how='left')
        data = data.rename(index=str, columns={0:f"{t}"})
        data.index = list(range(data.shape[0]))

    data.to_csv(csv_name)

    print("Ran in "+str(int(time.time()-start))+" seconds.")

    if remove_flicker:
        return data, num_of_diffs

    return data



def convert_to_gray(rawfilename, directory=None, is_job=False):

    params = get_params(rawfilename, directory, is_job)

    vc = cv.VideoCapture(params['directory']+rawfilename)

    success, frame = vc.read()

    height, width, layers = frame.shape
    fourcc = cv.VideoWriter_fourcc(*'FFV1')
    video_out = cv.VideoWriter(f'{params["directory"]}{params["videoname"]}_gray.avi', fourcc, 30, (width, height))

    while(success):

        bw_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        bw_frame = cv.cvtColor(bw_frame, cv.COLOR_GRAY2BGR)
        video_out.write(bw_frame)

        success, frame = vc.read()

    cv.destroyAllWindows()



def crop(rawfilename, directory=None, xtrim = 656, ytrim=0, is_job=False):

    params = get_params(rawfilename, directory, is_job)

    vc = cv.VideoCapture(params['directory']+rawfilename)

    success, frame = vc.read()
    print(frame.shape)

    height, width, layers = frame.shape
    fourcc = cv.VideoWriter_fourcc(*'FFV1')
    height -= ytrim*2
    width -= xtrim*2
    video_out = cv.VideoWriter(f'{params["directory"]}{params["videoname"]}_trimmed.avi', fourcc, 30, (width, height))
    print(f'{params["directory"]}{params["videoname"]}_trimmed.avi', fourcc, 30, (width, height))
    print(frame[:, xtrim:(xtrim+608), :].shape)
    while(success):
        video_out.write(frame[xtrim:(xtrim+608), :, :])
        success, frame = vc.read()

    cv.destroyAllWindows()







