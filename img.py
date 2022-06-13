import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as co
from numpy import fft
from scipy import stats

purple = "#621743ff"
blue = "#3d66afff"
orange = "#e26237ff"
green = "#7c8852ff"
gray = "#dadada"
lightorange = "#d5a078"
lightblue = "#909fd1"
darkorange = "#b94d17"
darkblue = "#213f76"
cmaplist = [darkblue, blue, lightblue, gray, lightorange, orange, darkorange]
cmap1 = co.LinearSegmentedColormap.from_list("mycmap", cmaplist)
cmapor = co.LinearSegmentedColormap.from_list("mycmap", cmaplist[3:])
cmapbl = co.LinearSegmentedColormap.from_list("mycmap", cmaplist[3:0:-1])
cmapor1 = co.LinearSegmentedColormap.from_list("mycmap", [orange, orange])
cmapbl1 = co.LinearSegmentedColormap.from_list("mycmap", [blue, blue])

class image: 
    def __init__(self, path, file, stretch): 
        self.img = None #numpy array representing image
        self.path = path 
        self.file = file
        self.stretch = stretch
        self.c = None #intensity of grey vignette (0 is black, 1 is white)
        self.dim = None #dimensions of image
        self.x = None #x-axis of image in pixels
        self.y = None #y-axis of image in pixels
        self.sum_x = None #sum of intensities along x (sum of rows)
        self.sum_y = None #sum of intensities along y (sum of columns)
        self.sp_x_uf = None #unfiltered FFT
        self.sp_y_uf = None #unfiltered FFT
        self.sp_x = None #FFT
        self.sp_y = None #FFT
        self.freq_x = None #FFT frequencies
        self.freq_y = None #FFT frequencies
        self.filt_x = None #filter
        self.filt_y = None #filter
        self.x_abs = None
        self.x_real = None
        self.x_imag = None
        self.y_abs = None
        self.y_real = None
        self.y_imag = None
        self.sum = None
        self.normal = True
        self.temp = None
        self.x_peaks = []
        self.y_peaks = []
        self.ft = None
        self.ftdim = None
        self.thresh = None
        self.box = 50

    def prepare_file(self): 
        img = cv2.imread(self.path + self.file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = cv2.resize(img, (int(2000*self.stretch)-1, 1999)) 
        self.dim = self.img.shape[:2]

    def add_vignette(self, vig, temp = False):
        img = np.array(self.img, dtype = np.uint32)
        img = img/255
        self.c = (np.max(img)-np.min(img))/2 + np.min(img)
        img = img-self.c
        if temp: 
            self.temp = img*vig
        else: 
            self.img = img*vig
    
    def sum_contrast(self, axis = [0, 1], temp = False): 
        if temp: 
            sum = [self.sum_x, self.sum_y]
            for i in axis:
                sum[i] = np.sum(self.temp, axis = i)/self.dim[i]
            self.sum_x, self.sum_y = sum
        else: 
            sum = [self.sum_x, self.sum_y]
            for i in axis:
                sum[i] = np.sum(self.img, axis = i)/self.dim[i]
            self.sum_x, self.sum_y = sum

    def save_window(self):
        img = (self.img + self.c)*255
        cv2.imwrite(self.path + "windows/" + self.file, img)

    def FFT(self, axis = [0, 1]):
        sum = [self.sum_x, self.sum_y]
        ax = [np.linspace(0, self.dim[1], self.dim[1]), np.linspace(0, self.dim[0], self.dim[0])]
        self.x, self.y = ax[0], ax[1]
        sp = [self.sp_x, self.sp_y]
        freq = [self.freq_x, self.freq_y]
        for i in axis: 
            sp[i] = fft.fftshift(fft.fft(sum[i]))
            freq[i] = fft.fftshift(fft.fftfreq(ax[i].shape[-1]))
        self.sp_x, self.sp_y = sp
        self.sp_x_uf, self.sp_y_uf = sp
        self.freq_x, self.freq_y = freq

    def filter(self, axis = [0, 1], center = 0.0204, sc = 0.001): # Ssc = 0.001
        freq = [self.freq_x, self.freq_y]
        sp = [self.sp_x, self.sp_y]
        filt = [self.filt_x, self.filt_y]
        for i in axis: 
            f =  stats.norm.pdf(freq[i], loc = center, scale = sc) #center = 0.0204 for simulated
            filt[i] = (f-np.min(f))/np.max(f) 
            sp[i] = np.multiply(sp[i], filt[i])
        self.sp_x, self.sp_y = sp
        self.filt_x, self.filt_y = filt

    def vals(self, axis = [0, 1]):
        abs = [self.x_abs, self.y_abs]
        real = [self.x_real, self.y_real]
        imag = [self.x_imag, self.y_imag]
        sp = [self.sp_x, self.sp_y]
        for i in axis: 
            abs[i] = np.sum(np.abs(sp[i]))
            real[i] = np.sum(sp[i].real)
            imag[i] = np.sum(sp[i].imag)
        self.x_abs, self.y_abs = abs
        self.x_real, self.y_real = real
        self.x_imag, self.y_imag = imag 
        self.sum = np.sum(abs)
    
    def norm(self, axis = [0, 1], f = None):
        abs = [self.x_abs, self.y_abs]
        real = [self.x_real, self.y_real]
        imag = [self.x_imag, self.y_imag]
        sp = [self.sp_x, self.sp_y]
        if f == None: 
            sumreal = np.sum(np.abs(real))
            sumimag = np.sum(np.abs(imag))
            for i in axis: 
                abs[i] = abs[i]/self.sum
                real[i] = real[i]/sumreal
                imag[i] = imag[i]/sumimag
        else: 
            for i in axis: 
                abs[i] = abs[i]/f   
                real[i] = real[i]/f
                imag[i] = imag[i]/f     
        self.x_abs, self.y_abs = abs
        self.x_real, self.y_real = real
        self.x_imag, self.y_imag = imag 

    def recipe(self, vig, f = None, center = 0.0204):
        self.prepare_file()
        self.add_vignette(vig)
        self.sum_contrast()
        self.save_window() 
        self.FFT()
        self.filter(center = center)
        self.vals()
        if self.normal:
            self.norm(f = f)

    def plot1D(self, filetype = 'svg'):
        #create plot, 6 subplots; 4 for contrast along x and y and their FFTs, 1 for original image and 1 for bar chart of peak heights
        fig, axs = plt.subplots(2, 3, gridspec_kw={'width_ratios': [4, 4, 3]})
        #fig.suptitle(file)
        for i in range(2): #editing ylimit of FFT-plots
            axs[1, i].set_ylim(-100, 100)
            axs[1, i].set_xlim(-0.025, 0.025)
            axs[0, i].grid()
            axs[1, i].grid()
            #axs[0, i].set_xlim(300, 1700)
            axs[0, i].set_ylim(-0.3, 0.3)

        #plot intensities in x and y (real space), and original image
        axs[0, 0].set_title("(a) Intensity along x")
        axs[0, 0].plot(self.x, self.sum_x)
        axs[0, 1].set_title("(b) Intensity along y")
        axs[0, 1].plot(self.y, self.sum_y)
        axs[0, 2].set_title("(c) Image")
        im = (self.img + self.c)*255
        axs[0, 2].imshow(im, cmap = "binary_r", vmin = 0, vmax = 255)
        axs[0, 2].set_yticks([])
        axs[0, 2].set_xticks([])
        #axs[0, 2].set_ylim(1700, 300)
        #axs[0, 2].set_xlim(300, 1700)

        axs[1, 0].plot(self.freq_x, np.abs(self.sp_x_uf), color = 'C0', alpha = 0.2)
        axs[1, 0].plot(self.freq_x, self.sp_x_uf.real, color = 'C1', alpha = 0.2)
        axs[1, 0].plot(self.freq_x, self.sp_x_uf.imag, color = 'C2', alpha = 0.2)
        
        axs[1, 0].set_title("(d) Frequency along x")
        axs[1, 0].plot(self.freq_x, np.abs(self.sp_x), label = "Abs")
        axs[1, 0].plot(self.freq_x, self.sp_x.real, label = "Re")
        axs[1,0].plot(self.freq_x, self.sp_x.imag, label = "Im")
        axs[1, 0].plot(self.freq_x, np.multiply(self.filt_x, 90), color = "gray", alpha = 1)
        axs[1, 0].legend(loc = 2)

        axs[1, 1].plot(self.freq_y, np.abs(self.sp_y_uf), color = 'C0', alpha = 0.2)
        axs[1, 1].plot(self.freq_y, self.sp_y_uf.real, color = 'C1', alpha = 0.2)
        axs[1, 1].plot(self.freq_y, self.sp_y_uf.imag, color = 'C2', alpha = 0.2)

        #plot
        axs[1, 1].set_title("(e) Frequency along y")
        axs[1, 1].plot(self.freq_y, np.abs(self.sp_y), label = "Abs")
        axs[1, 1].plot(self.freq_y, self.sp_y.real, label = "Re")
        axs[1, 1].plot(self.freq_y, self.sp_y.imag, label = "Im")
        axs[1, 1].plot(self.freq_y, np.multiply(self.filt_y, 90), color = "gray", alpha = 1)

        #last bar chart
        axs[1, 2].grid()
        axs[1, 2].set_title("(f) Integrated and\n normalized frequency")
        axs[1, 2].bar([-0.3, 0.7, 1.7], [self.x_abs, self.y_abs,self.x_abs + self.y_abs], width = 0.25, alpha = 1, label = "Abs")
        axs[1, 2].bar([0, 1, 2], [self.x_real, self.y_real, self.x_real + self.y_real], width = 0.25, alpha = 1, label = "Re")
        axs[1, 2].bar([0.3, 1.3, 2.3], [self.x_imag, self.y_imag, self.x_imag + self.y_imag], width = 0.25, alpha = 1, label = "Im")
        axs[1, 2].set_xticks([0, 1, 2])
        axs[1, 2].set_xticklabels(['x', 'y', "sum"])
        axs[1, 2].set_xlim(-0.5, 2.5)
        if self.normal: 
            axs[1, 2].set_ylim(-1, 1)

        #save plot  
        plt.subplots_adjust(hspace = 0.5, wspace = 0.3) 
        fig.savefig(self.path + "plots/" + self.file[:-3] + filetype)
        plt.cla()
        plt.close()
        del fig

    def dynamics(self, grid, vig_params, vig_blur, plot = True, center = 0.0204, f = None, cyclic = False, filetype = 'svg', printsum = False, abs = False, overlay = True):
        self.prepare_file()
        if type(grid) == list:
            gridx = int(grid[0])
            gridy = int(grid[1])
        else: 
            gridx, gridy = int(grid), int(grid)
        self.peaks_x = np.zeros((gridy, gridx), dtype = float)
        self.peaks_y = np.zeros((gridy, gridx), dtype = float)
        for i in range(gridx):
            for j in range(gridy):
                imgsizey = self.dim[0]-vig_params[0]-np.abs(vig_params[1])
                imgsizex = self.dim[1]-vig_params[2]-np.abs(vig_params[3])
                #print([int(vig_params[0] + j*imgsizey/grid),int(vig_params[0] + (j+1)*imgsizey/grid), int(vig_params[2] + i*imgsizex/grid),int(vig_params[0] + (i+1)*imgsizex/grid)])
                vig = vignette_params(self.stretch, [int(vig_params[0] + j*imgsizey/gridy),int(vig_params[0] + (j+1)*imgsizey/gridy), int(vig_params[2] + i*imgsizex/gridx),int(vig_params[0] + (i+1)*imgsizex/gridx)], vig_blur)
                self.add_vignette(vig, temp = True)
                self.sum_contrast(temp = True)
                self.FFT()
                if type(center) == float: 
                    self.filter(center = center)
                else:
                    self.filter(axis = [0], center = center[0])
                    self.filter(axis = [1], center = center[1])
                self.vals()
                if i%10 == 0:
                    if j%10 ==0:
                        if printsum: 
                            print(i, j, self.sum)
                if self.normal:
                    self.norm(f = f)
                if abs: 
                    self.peaks_y[j][i] = self.y_abs
                    self.peaks_x[j][i] = self.x_abs
                else:
                    self.peaks_y[j][i] = self.y_abs*np.sign(self.y_imag)*-1
                    self.peaks_x[j][i] = self.x_abs*np.sign(self.x_imag)*-1
        if plot: 
            fig, axs = plt.subplots(1, 3, figsize = (14, 4), gridspec_kw = {'height_ratios':[1]})
            fig.suptitle(self.file)
            if abs: 
                map = axs[2].imshow(self.peaks_y, vmin = 0, vmax = 1, cmap = cmapor, aspect = 'auto')
                plt.colorbar(map, ax = axs[2], fraction=0.046, pad=0.04)
                map2 = axs[1].imshow(self.peaks_x, vmin = 0, vmax = 1, cmap = cmapbl, aspect = 'auto')
                plt.colorbar(map2, ax = axs[1], fraction=0.046, pad=0.04)
            else: 
                map = axs[2].imshow(self.peaks_y, vmin = -1, vmax = 1, cmap = cmap1, aspect = 'auto')
                plt.colorbar(map, ax = axs[2], fraction=0.046, pad=0.04)
                axs[1].imshow(self.peaks_x, vmin = -1, vmax = 1, cmap = cmap1, aspect = 'auto')
            axs[1].set_title("Domains directed along y")
            axs[0].imshow(self.img, cmap = "binary_r", vmin = 0, vmax = 255, aspect = self.stretch)
            axs[0].set_title("Image")
            axs[2].set_title("Domains directed along x")
            for i in range(2):
                axs[i+1].set_ylim(gridy + 0.16*gridy, -0.32*gridy)
                axs[i+1].set_xlim((-0.32*gridx)*1/self.stretch, gridx + (0.16*gridx)*1/self.stretch)
            for i in range(3):
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            #save plot
            fig.savefig(self.path + "plots-dynamics/" + self.file[:-3] + filetype)
            plt.close()
            del fig
        if cyclic: 
            fig, axs = plt.subplots(1, 2, figsize = (9, 4)) #gridspec_kw = {"height_ratios":[1]}
            fig.suptitle(self.file)
            angs = np.array([[np.arctan2(self.peaks_x[j][i],self.peaks_y[j][i]) for i in range(grid)] for j in range(grid)], dtype = float)
            angs = np.array([[angs[j][i] + 2*np.pi if angs[j][i]<0 else angs[j][i] for i in range(grid)] for j in range(grid)])
            alphs = np.array([[np.sqrt(np.square(self.peaks_x[j][i]) + np.square(self.peaks_y[j][i])) for i in range(grid)] for j in range(grid)], dtype = float)
            map = axs[1].imshow(angs, vmin = 0, vmax = 2*np.pi, cmap = 'hsv', aspect = 'auto', alpha = alphs)
            #cbar = plt.colorbar(map, ax = axs[1], fraction=0.046, pad=0.04, ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            #cbar.ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
            axs[1].set_title("Domains")
            axs[0].imshow(self.img, cmap = "binary_r", vmin = 0, vmax = 255, aspect = self.stretch)
            axs[0].set_title("Image")
            axs[1].set_ylim(gridy + 0.32*gridy, -0.32*gridy)
            axs[1].set_xlim((-0.32*gridx)*1/self.stretch, gridx + (0.32*gridx)*1/self.stretch)
            for i in range(2):
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            fig.savefig(self.path + "plots-cyclic/" + self.file[:-3] + filetype)
            plt.close()
            del fig
        if overlay: 
            fig = plt.figure(figsize = (4*self.stretch, 4), frameon = False)
            plt.imshow(self.peaks_y, vmin = 0, vmax = 1, cmap = cmapor1, alpha = self.peaks_y, aspect = 1)
            plt.imshow(self.peaks_x, vmin = 0, vmax = 1, cmap = cmapbl1, alpha = self.peaks_x, aspect = 1)
            plt.ylim(gridy + 0.32*gridy, -0.32*gridy)
            plt.xlim((-0.32*gridx)*1/self.stretch, gridx + (0.32*gridx)*1/self.stretch)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            fig.savefig(self.path + "plots-overlay/" + self.file[:-3] + filetype)
            plt.close()
            del fig

    def FFT2D(self):
        im = (self.img + self.c)*255
        ft = fft.ifftshift(im)
        ft = fft.fft2(ft)
        ft = fft.fftshift(ft)
        self.ft = np.log(np.abs(ft))
        self.ftdim = self.ft.shape
    
    def threshold(self, thresh):
        t, ft, = cv2.threshold(self.ft, thresh, 255, cv2.THRESH_BINARY)
        self.thresh = ft
        return ft
    
    def mask(self, mask, thresh = True):
        if thresh:
            self.thresh = self.thresh*mask
        else: 
            self.thresh = self.ft*mask

    def counter(self):
        c1, c2 = 0, 0
        h, w = self.ftdim
        for x in range(int(w/2-self.box), int(w/2 + self.box)):
            for y in range(int(h/2 +self.box), int(h/2-self.box), -1):
                if self.thresh[y, x] > 0: 
                    if x > int(w/2):
                        if y < int(h/2):
                            c1 += self.thresh[y, x]
                        else: 
                            c2 += self.thresh[y, x]
        return c1, c2

    def plot2D(self, plotthresh = False, thresh = None, filetype = 'svg'):
        if plotthresh == False:
            fig, axs = plt.subplots(1, 3, figsize = (12, 4))
            fig.suptitle(self.file)
            im = (self.img + self.c)*255
            axs[0].imshow(im, cmap = "binary_r", vmin = 0, vmax = 255)
            axs[1].imshow(self.ft, cmap = "binary_r", vmin = -6, vmax = 20)
            axs[1].set_ylim(int(self.ftdim[0]/2+self.box+20), int(self.ftdim[0]/2 - self.box-20))
            axs[1].set_xlim(int(self.ftdim[1]/2-self.box-20), int(self.ftdim[1]/2 + self.box+20))
            ft = cv2.cvtColor(self.thresh.copy().astype("uint8"), cv2.COLOR_GRAY2BGR)
            ft = cv2.rectangle(ft, [int(self.ftdim[1]/2)-self.box, int(self.ftdim[0]/2)-self.box], [int(self.ftdim[1]/2)+self.box, int(self.ftdim[0]/2)+self.box], (255, 0, 0), 2)
            ft = cv2.line(ft, (int(self.ftdim[1]/2), int(self.ftdim[0]/2)-self.box), (int(self.ftdim[1]/2), int(self.ftdim[0]/2)+self.box), (255, 0, 0), 1)
            ft = cv2.line(ft, (int(self.ftdim[1]/2)-self.box, int(self.ftdim[0]/2)), (int(self.ftdim[1]/2)+self.box, int(self.ftdim[0]/2)), (255, 0, 0), 1)
            axs[2].imshow(ft, cmap = "binary_r", vmin = 0, vmax = 255)
            axs[2].set_ylim(int(self.ftdim[0]/2+self.box+20), int(self.ftdim[0]/2 - self.box-20))
            axs[2].set_xlim(int(self.ftdim[1]/2-self.box-20), int(self.ftdim[1]/2 + self.box+20))
        else: 
            fig, axs = plt.subplots(2, 2, figsize = (12, 12))
            fig.suptitle(self.file)
            im = (self.img + self.c)*255
            axs[0][0].imshow(im, cmap = "binary_r", vmin = 0, vmax = 255)
            axs[0][1].imshow(self.ft, cmap = "binary_r", vmin = -6, vmax = 20)
            axs[0][1].set_ylim(int(self.ftdim[0]/2+self.box+20), int(self.ftdim[0]/2 - self.box-20))
            axs[0][1].set_xlim(int(self.ftdim[1]/2-self.box-20), int(self.ftdim[1]/2 + self.box+20))
            ft = cv2.cvtColor(self.thresh.copy().astype("uint8"), cv2.COLOR_GRAY2BGR)
            ft = cv2.rectangle(ft, [int(self.ftdim[1]/2)-self.box, int(self.ftdim[0]/2)-self.box], [int(self.ftdim[1]/2)+self.box, int(self.ftdim[0]/2)+self.box], (226, 98, 55), 2)
            ft = cv2.line(ft, (int(self.ftdim[1]/2), int(self.ftdim[0]/2)-self.box), (int(self.ftdim[1]/2), int(self.ftdim[0]/2)+self.box), (226, 98, 55), 1)
            ft = cv2.line(ft, (int(self.ftdim[1]/2)-self.box, int(self.ftdim[0]/2)), (int(self.ftdim[1]/2)+self.box, int(self.ftdim[0]/2)), (226, 98, 55), 1)
            axs[1][1].imshow(ft, cmap = "binary_r", vmin = 0, vmax = 255)
            axs[1][1].set_ylim(int(self.ftdim[0]/2+self.box+20), int(self.ftdim[0]/2 - self.box-20))
            axs[1][1].set_xlim(int(self.ftdim[1]/2-self.box-20), int(self.ftdim[1]/2 + self.box+20))
            axs[1][0].imshow(thresh, cmap = "binary_r", vmin = 0, vmax = 255)
            axs[1][0].set_ylim(int(self.ftdim[0]/2+self.box+20), int(self.ftdim[0]/2 - self.box-20))
            axs[1][0].set_xlim(int(self.ftdim[1]/2-self.box-20), int(self.ftdim[1]/2 + self.box+20))

        fig.savefig(self.path + "2D-FFT/" + self.file[:-3] + filetype)
        plt.close()

    def recipe2D(self, vig, thresh, msk):
        self.prepare_file()
        self.add_vignette(vig)
        self.FFT2D()
        t = self.threshold(thresh)
        self.mask(msk)
        c1, c2 = self.counter()
        self.plot2D(plotthresh = True, thresh = t)
        return c1, c2

def mask(slope, stretch, box = 5):
    mask = np.zeros((1999, int(2000*stretch)-1), np.uint8)
    h, w = mask.shape
    for x in range(w):
        for y in range(h):
            if abs(int(y-int(h/2)))>abs(int(slope*(x-int(w/2)))):
                mask[y][x] = 255
            if abs(int(x-int(w/2)))>abs(int(slope*(y-int(h/2)))):
                mask[y][x] = 255
    mask = cv2.bitwise_not(mask)
    for x in range(int(w/2)-box, int(w/2)+1+box):
        for y in range(int(h/2)-box, int(h/2)+1+box):
            mask[y][x] = 0
    cv2.imwrite("mask%i.jpg"%(10*stretch), mask)
    return mask/255

def vignette_params(stretch, vig_params, vig_blur, save = False, savepath = None):
    vignette = np.zeros((1999, int(2000*stretch)-1), np.uint8)
    vignette[vig_params[0]:vig_params[1], vig_params[2]:vig_params[3]] = 255
    vignette = cv2.GaussianBlur(vignette, (vig_blur, vig_blur), 0)
    if save:
        cv2.imwrite(savepath + "vignette.png", vignette)
    return vignette/255
