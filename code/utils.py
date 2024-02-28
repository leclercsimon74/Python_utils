# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:44:14 2021

@author: sleclerc
Last updated one

"""

import numpy as np
from skimage import filters, measure
import re
import tifffile
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import math
from itertools import cycle
from scipy.ndimage.filters import gaussian_filter


#image function
def get_img(filename):
    img = np.array(tifffile.imread(filename))
    if np.min(img) < 0: #if warp, a 8 bit image only... Should NOT happen
        img = np.where(img<0, img+256, img)
    return img


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    Should not print anything else than this bar during the progression!
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
    

def found_img(path, extension='.tif'):
    """
    Found all images with the file extension. Can remplace the file extension
    with a search string, such as 'mock cell 001' will found all image with
    this in its name /b in the folder b/. This is not recursive, meaning that 
    it will not search in subfolder.
    Parameters
    ----------
    path : str
        local or global string path
    extension : str, optional
        A search string option. The default is '.tif'. '' will deactivate it.

    Returns
    -------
    img_path : list
        a list of string indicate where to found each selected img

    """
    imgs = [f for f in os.listdir(path) if f.endswith(extension)]
    imgs_path = [path+"\\"+f for f in imgs]
    return imgs_path


def recursive_found_img(path, extension='.tif'):
    """
    Like found_img, but recursive.

    Parameters
    ----------
    path : str
        local or global string path
    extension : str, optional
        A search string option. The default is '.tif'. '' will deactivate it.

    Returns
    -------
    l : list
        a list of string indicate where to found each selected img

    """
    l = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                l.append(root+os.sep+file)
    return l


def norm(arr):
    #normalize an array 0-1
    return (arr - np.min(arr))/np.ptp(arr)


def meanZ_graph(R, G, B, img_name, multi=(1,1,0.33), label=['','','Nucleus']):
    """
    False 3D visualization. Each channel should be a 3D image binarize.
    A function will remplace all value higher than 0 by the corresponding
    multiplicator value, then realize a mean calculation along the z axis. 
    This mean that the signal intensity is proportional to the thickness.
    By convention, the nucleus is Blue, and to avoid a too strong signal,
    its multiplicator should be smaller that 1.
    Parameters
    ----------
    R : 3D numpy array
        Binarized or labeled image.
    G : 3D numpy array
        Binarized or labeled image.
    B : 3D numpy array
        Binarized or labeled image.
    img_name : str
        Add to the title of the image for identification.
    multi : tuple, optional
        Corresponding multiplicator value for the different channel. The default is (5,2,0.33).
    label : list of str
        list if label for the legend. The default is ['Nucleolus','NS2','Nucleus'].
    
    Returns
    -------
    None.

    """
    #Show 2D results
    fig = plt.figure(figsize=(5,5))
    #Create a RGB image, where all value higher than 0 are considered
    RGB = np.dstack(np.array((np.mean(np.where(R > 0, multi[0], 0), axis=0),
                              np.mean(np.where(G > 0, multi[1], 0), axis=0),
                              np.mean(np.where(B > 0, multi[2], 0), axis=0))))
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=[1,0,0], label=label[0]),
               mpatches.Patch(color=[0,1,0], label=label[1]),
               mpatches.Patch(color=[0,0,1], label=label[2])]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, loc='best', borderaxespad=0., framealpha=1)
    plt.title('Mean z axis of '+img_name)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(RGB)
    fig.set_facecolor('w')
    plt.show()


def maxZ_graph(R, G, B, img_name, multi=(1,1,0.33), label=['','','Nucleus']):
    """
    False 3D visualization. Each channel should be a 3D image binarize.
    A function will remplace all value higher than 0 by the corresponding
    multiplicator value, then realize a mean calculation along the z axis. 
    This mean that the signal intensity is proportional to the thickness.
    By convention, the nucleus is Blue, and to avoid a too strong signal,
    its multiplicator should be smaller that 1.
    Parameters
    ----------
    R : 3D numpy array
        Binarized or labeled image.
    G : 3D numpy array
        Binarized or labeled image.
    B : 3D numpy array
        Binarized or labeled image.
    img_name : str
        Add to the title of the image for identification.
    multi : tuple, optional
        Corresponding multiplicator value for the different channel. The default is (5,2,0.33).
    label : list of str
        list if label for the legend. The default is ['Nucleolus','NS2','Nucleus'].
    
    Returns
    -------
    None.

    """
    #Show 2D results
    fig = plt.figure(figsize=(5,5))
    #Create a RGB image, where all value higher than 0 are considered
    RGB = np.dstack(np.array((np.max(R, axis=0),
                              np.max(G, axis=0),
                              np.max(B, axis=0))))
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=[1,0,0], label=label[0]),
               mpatches.Patch(color=[0,1,0], label=label[1]),
               mpatches.Patch(color=[0,0,1], label=label[2])]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, loc='best', borderaxespad=0., framealpha=1)
    plt.title('Max z axis of '+img_name)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(to_8bits(RGB))
    fig.set_facecolor('w')
    plt.show()
    
    
def ortho_view(img, kind, title):
    """
    Generate an ortho view of the 3D img. Can choose between a mean and max view.

    Parameters
    ----------
    img : 3D numpy array
        3 dimensions numpy array (ZXY)
    kind : str
        Kind of projection, only 'mean' and 'max'. mean by default/error
    title : str
        Graph title if any. '' will not add a title

    Returns
    -------
    None.

    """
    if len(img.shape) != 3:
        print('error, bad shape!')
        return
    
    if kind == 'max':
        top = np.max(img, axis=0)
        side = np.rot90(np.max(img, axis=2), k=-1)
        bottom = np.max(img, axis=1)
    else:
        top = np.mean(img, axis=0)
        side = np.rot90(np.mean(img, axis=2), k=-1)
        bottom = np.mean(img, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 10) , nrows=2, ncols=2,
                           gridspec_kw={'width_ratios': [10, 1],
                                        'height_ratios': [10, 1]})
    
    for a in ax: #remove all axis
        for b in a:
            b.set_axis_off()
    
    ax[0,0].imshow(top)
    ax[0,1].imshow(side)
    ax[1,0].imshow(bottom)
    
    plt.tight_layout() 
    if title != '':
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=0.95)
    plt.show()


def ortho_view_slice(img, title='', z_focus=None, y_focus=None, x_focus=None, color='w', alpha=.3):
    """
    View of a slice of a 3D image

    Parameters
    ----------
    img : np.array
        3D array (numpy)
    title : str, optional
        Title of the image if needed. The default is ''.
    z_focus : int, optional
        Where to focus in z. The default is half.
    y_focus : int, optional
        Where to focus the y. The default is half.
    x_focus : int, optional
        Where to focus the x. The default is half.
    color : str, optional
        Matplotlib color of the position indicator. The default is 'w'.
    alpha : float, optional
        Alpha value of the position indicator. Between 0 and 1. The default is .3.

    Returns
    -------
    None.

    """
    if title == '':
        title = 'Ortho-view'

    shape = img.shape
    #default
    if z_focus is None:
        z_focus = shape[0]//2
    if x_focus is None:
        x_focus = shape[1]//2
    if y_focus is None:
        y_focus = shape[2]//2
    
    top = img[z_focus]
    side = np.flipud(np.rot90(img[:,:,x_focus])) #some mod to fit the orientation
    bottom = img[:,y_focus,:]
    
    fig, ax = plt.subplots(figsize=(10, 10) , nrows=2, ncols=2,
                           gridspec_kw={'width_ratios': [2, 1],
                                        'height_ratios': [2, 1]})
    
    ax[0,0].imshow(top, aspect='equal')
    ax[0,0].plot((0, shape[2]-1),(y_focus, y_focus), color, alpha=alpha)
    ax[0,0].plot((x_focus, x_focus),(0, shape[1]-1), color, alpha=alpha)
    ax[0,0].set_ylabel('y')
    ax[0,0].xaxis.set_ticks_position('top')
    ax[0,0].xaxis.set_label_position('top')
    ax[0,0].set_xlabel('x')
    
    ax[0,1].imshow(side, aspect='auto')
    ax[0,1].plot((z_focus, z_focus), (0, shape[1]-1), color, alpha=alpha)
    ax[0,1].plot((0, shape[0]-1), (y_focus, y_focus), color, alpha=alpha)
    ax[0,1].set_xlabel('z')
    ax[0,1].xaxis.set_ticks_position('top')
    ax[0,1].xaxis.set_label_position('top')
    ax[0,1].get_yaxis().set_ticks([])
    
    ax[1,0].imshow(bottom, aspect='auto')
    ax[1,0].plot((0, shape[2]-1),(z_focus, z_focus), color, alpha=alpha)
    ax[1,0].plot((x_focus, x_focus), (0, shape[0]-1), color, alpha=alpha)
    ax[1,0].set_ylabel('z')
    ax[1,0].get_xaxis().set_ticks([])
    
    ax[1,1].set_axis_off()
    
    plt.tight_layout() 
    
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(top=0.92)
    plt.show()


def generate_spiral(img_name, c, overlap=10, save=True, visu=True):
    """
    Assemble a spiral serie from Leica to a smoothed uniformized single channel image.
    Extra parameters can be found in the core body:
        - gaussian blur
        - overlap threshold
        - signal threshold
        - compensating noise value

    Parameters
    ----------
    img_name : str
        image name or path to open a tiff image from a Leica spiral.
    c : TYPE
        Channel to choose.
    overlap : int, optional
        % of overlap of the image. The default is 10.
    save : bool, optional
        Save the result image as tiff, including the channel as c01. The default is True.
    visu : bool, optional
        Show the image normalization and end result. The default is True.

    Returns
    ------
    dest: numpy array
        Reconstructed spiral image

    """
    #read the img
    img = tifffile.imread(img_name)
    
    #get the number of img composing it
    print('Image shape is '+str(img.shape))
    n = img.shape[0]
    n_x = img.shape[2]
    n_y = img.shape[3]

    #Overlap settings
    o_x = int(n_x//overlap)
    o_y = int(n_y//overlap)
    o_thr = 1 #threshold value
    
    #size of the grid, by edges
    grid_size = int(math.ceil(math.sqrt(n)))
    
    #destination 2D array. oversize!
    dest = np.zeros((grid_size*n_x, grid_size*n_y), dtype=int)
    #reducing from the overlap
    dest = np.zeros((grid_size*(n_x-o_x)+o_x, grid_size*(n_y-o_y)+o_y), dtype=int)
    
    #origin, center of the grid
    ori = int(grid_size // 2), int(grid_size // 2)
    
    #move functions and order
    def up(x,y):
        return x, y+1
    def right(x,y): 
        return x+1, y
    def down(x,y):
        return x, y-1
    def left(x,y):
        return x-1, y
    moves = [left, up, right, down] #correct move order
    
    def gen_points(end):
        _moves = cycle(moves)
        i = 0
        pos = ori
        time_to_move = 1
        
        yield i, pos
        
        while True:
            for _ in range(2):
                move = next(_moves)
                for _ in range(time_to_move):
                    if i >= end:
                        return
                    pos = move(*pos)
                    i += 1
                    yield i, pos
            time_to_move += 1
                
    spiral = list(gen_points(n-1)) #generate list of point position
    
    g = 9 #gaussian value
    smooth = np.where(img[:,c,:,:] > 10, 2, img[:,c,:,:]) #replace signal
    smooth = gaussian_filter(smooth, (g,g,g))
    smooth = np.mean(smooth, axis=0)
    smooth += 1 #to avoid divide by 0
    smooth_mean = np.mean(smooth)
    
    for i in spiral:  
        x_s = int(i[1][0]*(n_x-o_x))
        x_e = int(i[1][0]*(n_x-o_x)+n_x)
        y_s = int(i[1][1]*(n_y-o_y))
        y_e = int(i[1][1]*(n_y-o_y)+n_y)
        p = dest[x_s:x_e, y_s:y_e]
        im = img[i[0],c,:,:]
        #illumination correction
        correct_im = im / smooth
        correct_im = np.array(correct_im * smooth_mean, dtype=int)
        #transform value below threshold to nan
        p = np.where(p>o_thr, p, np.nan)
        im = np.where(correct_im>o_thr, correct_im, np.nan)
        #average them
        mean = np.nanmean([p, im], axis=0)
        mean = np.nan_to_num(mean) #retransform nan to 0
        #place
        dest[x_s:x_e, y_s:y_e] = mean
    

    save_file = img_name.replace('.tif', '')
    save_file = save_file+'_c'+str(c).zfill(2)+'.tif'
    if save: tifffile.imsave(save_file, data=dest)
    if visu:#visu check
        plt.imshow(smooth)
        plt.title('Image normalization')
        plt.axis('off')
        plt.show()
        
        plt.figure(figsize=(grid_size,grid_size))
        plt.imshow(dest, interpolation=None)
        plt.axis('off')
        plt.show()

    return dest

    
#segmentation
def thr_list(data, thr):
    """
    Return the threshold value of the threshold method applied on data
    Parameters
    ----------
    data : array
        array, whatever is the dimension (flatten).
    thr : string
        Threshold name. minimum, yen, otsu, mean, triangle, isodata and li for the moment.

    Returns
    -------
    thr_value : int or float
        Value of the thresholded data.

    """
    data = np.ndarray.flatten(data)
    if thr == 'minimum':
        thr_min = filters.threshold_minimum(data)
    if thr == 'yen':
        thr_min = filters.threshold_yen(data)
    if thr == 'otsu':
        thr_min = filters.threshold_otsu(data)
    if thr == 'mean':
        thr_min = filters.threshold_mean(data)
    if thr == 'triangle':
        thr_min = filters.threshold_triangle(data)
    if thr == 'isodata':
        thr_min = filters.threshold_isodata(data)
    if thr == 'li':
        thr_min = filters.threshold_li(data)
    
    return thr_min


def measure_stats(binary, img):
    """
    Get some parameters for measurment, using the regionprops from scikit

    Parameters
    ----------
    nucleolin_binary : 3D numpy array
        The segmented and labeled nucleolin 3D image.
    img : 3D numpy array
        The channel to measure intensity from
    img_name: str
        Path of the image from which the analysis is done

    Returns
    -------
    props : Dataframe
        Pandas dataframe, use the concatenate function to merge with other.
        Add the path of the file as 'path'.

    """
    binary = measure.label(binary, background = 0)
    
    to_measure = ['label', 'area', 'centroid', 'convex_area', 'euler_number',
                  'inertia_tensor', 'major_axis_length', 'minor_axis_length',
                  'max_intensity', 'mean_intensity','min_intensity', 'solidity',
                  'moments_central', 'bbox']
    
    to_measure2D = ['label', 'area', 'centroid', 'convex_area', 'euler_number',
                  'inertia_tensor', 'major_axis_length', 'minor_axis_length',
                  'max_intensity', 'mean_intensity','min_intensity', 'solidity',
                  'orientation', 'feret_diameter_max','eccentricity','moments_central', 'bbox']
    
    if len(binary.shape) == 3:
        props = pd.DataFrame.from_dict(measure.regionprops_table(binary, img,
                                                            properties=to_measure))
    elif len(binary.shape) == 2:
        props = pd.DataFrame.from_dict(measure.regionprops_table(binary, img,
                                                            properties=to_measure2D))
    
    return props


def vertical_best_count(matrices, n=5, graph=False):
    """
    Mask a 3D array of bool with the continuous value in Z higher than n.
    original: https://stackoverflow.com/questions/44439703/python-find-consecutive-values-in-3d-numpy-array-without-using-groupby
    Parameters
    ----------
    matrices : 3D array of bool
        numpy array to get the best count
    n : int, optional
        Number of coninuous value. The default is 5.
    graph : bool, optional
        To display the result in false 3d. The default is False.

    Returns
    -------
    results : 3D array of bool
        Same shape than original, but with value not corresping removed

    """
    bests = np.zeros(matrices.shape[1:])
    counter = np.zeros(matrices.shape[1:])
    
    for depth in range(matrices.shape[0]):
        this_level = matrices[depth, :, :]
        counter = counter + this_level
        bests = (np.stack([bests, counter], axis=0)).max(axis=0)
    
    results = np.zeros(matrices.shape)
    for depth in range(matrices.shape[0]):
        this_level = matrices[depth, :, :]
        results[depth] = np.logical_and(bests > n, this_level)
    
    if graph:
        plt.imshow(np.sum(results, axis=0), interpolation='none')
        plt.title('results with n='+str(n))
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.show()
    
    return results


def measure_curvature(img, res, step=1, individual=True):
    """
    Measure the curvature of an 3D segmented object. This require as input a 
    binary 3D image. This use a python adapted version of a blender code:
    https://blender.stackexchange.com/questions/201804/how-calculate-the-surface-curvature-for-each-vertex-of-a-mesh
    It is possible to visualize the curvature and 3D object using open3D.
    This code use the marching cube surface creation from scikit image,  from 
    which the resolution is needed to give an accurate representation
    of the 3D volume. The step is by default 1, meaning that the surface is 
    created at 1 voxel resolution. This algorithm is slow, since it processes
    each vertex of the surface by founding its neighboors before calculating
    the curvature of this vextex. There is a progress bar to estimate the time
    before completion.

    Parameters
    ----------
    img : 3D numpy array
        A binary array describing the object (one or multiple) from wich the 
        curvatures will be measured.
    res : list or tuple or array of float
        The resolution. Should be in the xyz order.
    step : int, optional
        The resolution of the marching cube algorithm. The default is 1.
    individual : bool, optional
        To decide to merge all the measure in one object or separate them. The
        default is True.

    Returns
    -------
    TYPE, list
        Return 3 lists:
            1. a list of list containing the curvatures of each vertex
            2. a list of the mean curvature of each object
            3. a list of the standard deviation of the curvature of each object

    """
    from skimage import measure
    from math import asin
    
    def search_link(value, links, passed):
        for link in links:
            for l in link:
                if l in passed: continue
                else: break
            if value in link: return link
        return None
    
    def link_faces(idx):
        linked = []
        for f in faces:
            if idx in f:
                linked.append(f)
        return linked
    
    def ring_from_vert(idx): #idx is the vertex id
        vertices = link_faces(idx)
        vertices = [list(x) for x in vertices]
        passed = [idx]
        uni = list(np.unique(vertices))
        uni = [x for x in uni if x not in passed]
        result = [uni[0]]
        passed.append(uni[0])
        c = 0 #security counter to avoid any endless loop
        while len(result) < len(uni) and c<100:
            c += 1
            prev = search_link(result[-1], vertices, passed)
            if prev is not None:
                #inter = list(set(result[-1]) & set(prev))
                inter = [x for x in prev if x not in passed]
                if len(inter) > 0:
                    passed.append(inter[0])
                    result.append(inter[0])
                    vertices.remove(list(prev))
        return result #coo and index of the ring vertx
    
    
    def curvature_along_edge(vert, other_idx):
        normal_diff = normals[other_idx] - normals[vert]
        vert_diff = verts[other_idx] - verts[vert]
        return np.dot(normal_diff, vert_diff) / np.dot(vert_diff, vert_diff)
    
    
    def angle_between_edges(vert, other1, other2):
        edge1 = other1 - vert
        edge2 = other2 - vert
        product = np.cross(edge1, edge2)
        sinus = np.linalg.norm(product) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))
        return asin(min(1.0, sinus))
    
    
    def mean_curvature_vert(vert, idx):
        ring_idx = ring_from_vert(idx)
        ring_curvatures = []
        for x, r in enumerate(ring_idx):
            ring_curvatures.append(curvature_along_edge(idx, ring_idx[x]))
        
        total_angle = 0.0
        curvature = 0.0
        for i in range(len(ring_idx)-1):
            angle = angle_between_edges(verts[idx], verts[ring_idx[i]], verts[ring_idx[i+1]])
            total_angle += angle
            curvature += angle * (ring_curvatures[i] + ring_curvatures[i+1])
        
        return curvature / (2.0 * total_angle)
    
    
    def mean_curvature(vertices):
        curvatures = []
        print('')
        for idx, vert in enumerate(vertices):
            curvature = mean_curvature_vert(vert, idx)
            curvatures.append(curvature)
            printProgressBar(idx+1, len(vertices), prefix='Progress:', suffix='', length=45)
        return np.array(curvatures)
    
    #main running piece of code for this function
    if individual: #determine if running for each nucleoli or not
        label = measure.label(img)
        img = []
        for x in np.unique(label):
            if x == 0: continue #background value
            img.append(np.where(label==x, 255, 0))
    else:
        if np.max(img) == 1:
            img = np.where(img==1, 255, 0)
        img = [img]
    
    std_curves = []
    mean_curves = []
    all_curves = []
    for i in img: #calculating the curbature. Takes time!
        verts, faces, normals, values = measure.marching_cubes(i, 1, spacing=res,
                                                               step_size=step, allow_degenerate=False)
        curves = mean_curvature(verts)
        print('mean curvature: '+str(np.mean(curves)))
        print('std curvature: '+str(np.std(curves)))
        std_curves.append(np.std(curves))
        mean_curves.append(np.mean(curves))
        all_curves.append(curves)
    return all_curves, mean_curves, std_curves #Not sure about this one. So everything!
    

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def boolean_remove_zeros(arr):
    """Mini function to remove all zeros from an array"""
    return arr[arr != 0]


def to_8bits(array):
    """Normalize an array between 0 and 1 before transforming to a 8 bit"""
    return np.array(norm(array)*255, dtype='uint8')


def Zto_RGB(array):
    """Simple function to make a max projection of a 3D multichannel to a 2D
    matplotlib readable format."""
    RGB = np.dstack(np.array((np.max(array[2], axis=2),
                              np.max(array[1], axis=2),
                              np.max(array[0], axis=2))))
    
    return np.array(RGB, dtype='uint8')


def to_RGB(array):
    """Simple function to transform an 2D multichannel image in a 2D matplotlib
    readable format"""
    if array.shape[0] == 3:
        RGB = np.dstack((array[2], array[1], array[0]))
    elif array.shape[0] == 1:
        RGB = np.dstack((array[0], array[0], array[0]))
    else:
        print(array.shape)
    return np.array(RGB, dtype='uint8')


def img_color_merger(r=[], g=[], b=[], gr=[], c=[], m=[], y=[]):
    """
    Merge color channel together to obtain a RGB image for plotting (x, y, c).
    Merging is done by maximum choices.
    Tested only with 2D array.
    Parameters
    ----------
    r : 2D array, optional
        DESCRIPTION. The default is [].
    g : 2D array, optional
        DESCRIPTION. The default is [].
    b : 2D array, optional
        DESCRIPTION. The default is [].
    gr : 2D array, optional
        DESCRIPTION. The default is [].
    c : 2D array, optional
        DESCRIPTION. The default is [].
    m : 2D array, optional
        DESCRIPTION. The default is [].
    y : 2D array, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    3D array int8
        RGB merged color array, in the x,y,3 shape

    """
    a = [r,g,b,gr,c,m,y]
    for color in a: #get the shape of the image
        if len(color) != 0:
            blend = np.zeros((3,)+color.shape)
    #classical RGB
    if len(r) != 0:
        blend[0] = r
    if len(g) != 0:
        blend[1] = g
    if len(b) != 0:
        blend[2] = b
    #gray
    if len(gr) != 0:
        blend = np.maximum.reduce([blend, np.array([gr, gr, gr])])
    #less classical CMY color
    if len(c) != 0:
        c = np.array([np.zeros(c.shape), c, c])
        blend = np.maximum.reduce([blend, c])
    if len(m) != 0:
        m = np.array([m, np.zeros(m.shape), m])
        blend = np.maximum.reduce([blend, m])
    if len(y) != 0:
        y = np.array([y, y, np.zeros(y.shape)])
        blend = np.maximum.reduce([blend, y])
    #transform in matplotlib readable format
    blend = np.dstack((blend[0], blend[1], blend[2]))
    return np.array(blend, dtype='uint8')


#plot function
def get_color(nb, kind, dic=False):
    """
    Get a list of colors (RGB 0-1) of lenght nb. Two style, bars and points.
    Actually have 8 differents colors, but cycle through them to generate 
    enough points. Based on https://healthdataviz.com/2012/02/02/optimal-colors-for-graphs/

    Parameters
    ----------
    nb : int
        The number of color required
    kind : str
        Choose between bars and points (if empty/invalid, return points)
    dic : bool, optional
        If True, return the base dictionnary with the color value (len(8)). The default is False.

    Returns
    -------
    colors : list
        List of 3-tuples (RGB) of lenght nb.

    """
    bars = {'blue':(114/255, 147/255, 203/255),
           'orange':(225/255, 151/255, 76/255),
           'green':(132/255, 186/255, 91/255),
           'red':(211/255, 94/255, 96/255),
           'grey':(128/255, 133/255, 133/255),
           'purple':(144/255, 103/255, 167/255),
           'bordeau':(171/255, 104/255, 87/255),
           'gold':(204/255, 194/255, 16/255)
        }

    points = {'blue':(57/255, 106/255, 177/255),
           'orange':(218/255, 124/255, 48/255),
           'green':(62/255, 150/255, 81/255),
           'red':(204/255, 37/255, 41/255),
           'grey':(83/255, 81/255, 84/255),
           'purple':(107/255, 76/255, 154/255),
           'bordeau':(146/255, 36/255, 40/255),
           'gold':(148/255, 139/255, 61/255)
        }
    
    if kind == 'bars':
        colors = bars
    else:
        colors = points
    
    if dic: return colors
    
    nb = int(nb) #security
    if nb <= len(colors):
        colors = list(colors.values())[:nb]
    else:
        colors = [list(colors.values()) for x in range(int(np.ceil((nb/len(colors)))))]
        colors = [item for sublist in colors for item in sublist]
    
    return colors
