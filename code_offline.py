code_scan = """
import os
import requests
import re

progress_var.set(0)

files = os.listdir(folder_path_input)
new_files = []
for file in files:
    if file.split('.')[-1] in ['png', 'jpg', 'PNG', 'JPG', 'JPEG', 'jpeg']:
        new_files.append(file)
files = new_files

clear_folder(folder_path_output + '/temp/')
for i in tqdm.tqdm(range(len(files))):
    if i == 1:
        load_res(0)
    
    file = files[i]
    
    progress_var.set(int(((i+1)/len(files))*100))
    top.update_idletasks()

    img = cv2.imread(folder_path_input + '/' + file)

    if img.shape[-1] != 3:
        img = img[:,:,:3]

    mask, mask_refined, blk_list = dispatch(img, use_cuda)
    if use_cuda:
        torch.cuda.empty_cache()

    mask = cv2.dilate((mask > 170).astype('uint8')*255, np.ones((5,5), np.uint8), iterations=5)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    filter_mask = np.zeros_like(mask)
    for i, blk in enumerate(blk_list):
        xmin, ymin, xmax, ymax = blk.xyxy
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1

    bboxes = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        # index = np.bincount(np.ravel(filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)])).argmax()
        index = np.sum(filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)])
        if index > 0:
            bboxes.append(list(bbox))
            filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 0

    texts = []
    for bbox in bboxes:
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        text = mocr(img[int(ymin):int(ymax), int(xmin):int(xmax), :])
        if use_cuda:
            torch.cuda.empty_cache()
        texts.append(text)

    frames = [[0, img.shape[0],int(img.shape[1]/2), img.shape[1]], [0, img.shape[0], 0, int(img.shape[1]/2)]]
    frame_img = np.zeros_like(mask)
    frame_boxes = []
    frame_texts = []

    for i, frame in enumerate(frames):
        ymin, ymax, xmin, xmax = frame
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        frame_img[ymin: ymax, xmin:xmax] = i+1
        frame_boxes.append([])
        frame_texts.append([])

    for bbox, text in zip(bboxes,texts):
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        index = np.bincount(np.ravel(frame_img[int(ymin):int(ymax), int(xmin):int(xmax)])).argmax()
        if index > 0:
            frame_boxes[index-1].append(bbox)
            frame_texts[index-1].append(text)

    final_text = []
    final_bboxes = None
    for _bboxes, _texts in zip(frame_boxes, frame_texts):
        if len(_bboxes) != 0:
            a = np.array(_bboxes)
            arg =  np.argsort(a[:,1])
            # arg = np.argsort(img.shape[1] - (a[:,0] + a[:,2]))
            # arg1 =  np.argsort(a[:,1])
            # arg = np.argsort(np.argsort(arg)*np.argsort(arg1)*(img.shape[1] - a[:,0])*(a[:,1]))
            _texts = np.array(_texts)[arg.astype(int)]
            final_text.append(separator.join(_texts))
            if final_bboxes is None:
                final_bboxes = a[arg.astype(int)]
            else:
                final_bboxes = np.concatenate((final_bboxes, a[arg.astype(int)]))

    
    text = separator.join(final_text)
    text_ref = separator.join(final_text)
    text_ref = re.sub(re_str, '', text_ref)
    if not text_ref == "":
        with open(folder_path_output + '/' + file.split('.')[0] + '.txt', 'w+', encoding="utf-8") as f:
            f.write(text)
        f.close()
        with open(folder_path_output + '/temp/' + file.split('.')[0] + '.txt', 'w+', encoding="utf-8") as f:
            f.write(text)
        f.close()
        np.savetxt(folder_path_output + '/temp/' + file.split('.')[0] + '_bbox.txt', final_bboxes.astype(int))
        np.savetxt(folder_path_output + '/temp/' + file.split('.')[0] + '_order.txt', np.array(range(len(final_bboxes))).astype(int))
"""