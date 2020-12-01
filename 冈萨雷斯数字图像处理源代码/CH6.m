%% 第六章 彩色图像处理

%% rgbcube patch
clc
clear
rgbcube(-10,-10,4)
title('RGB 立方体')


%% imapprox 用较少的颜色来近似一幅索引图像
clc
clear
[X, map] = imread('trees.tif'); % max(X(:)) = 127
imshowMy(X, map)
title('原始索引彩色图像（128色）')
imfinfoMy(X)

[Y, newmap] = imapprox(X, map, 64);
imshowMy(Y, newmap)
title('索引图像（64色）')
imfinfoMy(Y)

[Y, newmap] = imapprox(X, map, 32);
imshowMy(Y, newmap)
title('索引图像（32色）')

[Y, newmap] = imapprox(X, map, 16);
imshowMy(Y, newmap)
title('索引图像（16色）')

[Y, newmap] = imapprox(X, map, 8);
imshowMy(Y, newmap)
title('索引图像（8色）')

[Y, newmap] = imapprox(X, map, 4);
imshowMy(Y, newmap)
title('索引图像（4色）')

[Y, newmap] = imapprox(X, map, 2);
imshowMy(Y, newmap)
title('索引图像（2色）（非黑白）')


%% dither
clc
clear
RGB  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
imshowMy(RGB)
title('原始真彩色（256*256*256色）图像')

BW = dither(rgb2gray(RGB));
imshowMy(BW)

GRAY = rgb2gray(RGB);
imshowMy(GRAY)
title('灰度[亮度]（256色）图像')

%% dither
clc
clear
RGB  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
imshowMy(RGB)
title('原始真彩色（256*256*256色）图像')

[IND map] = rgb2ind(RGB,256);
imshow(IND, map)
X = dither(RGB,map);
imview(X)

%% dither
clc
clear
RGB  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');

X = dither(RGB,jet(256));
imview(X)


%% dither
clc
clear
I = imread('cameraman.tif');
BW = dither(I);
imview(BW)

%% grayslice
clc
clear

I = imread('snowflakes.png');
imshowMy(I)
title('原始灰度（256色）图像')

X = grayslice(I,16);
imshowMy(X,jet(16))

%% ind2gray
clc
clear
load trees
I = ind2gray(X,map);   % max(X(:))=128
imshowMy(X,map)
title('原始索引图像')

imshowMy(I)    % max(I(:))=1
title('索引图像转化为灰度图像')

%% rgb2ind
clc
clear
RGB = imread('peppers.png');
imshowMy(RGB)
title('原始真彩色（256*256*256色）图像')

[X,map] = rgb2ind(RGB,128);
imshowMy(X,map)
title('彩色图像转化为索引图像')

%% ind2rgb 
clc
clear
RGB = imread('peppers.png');
imshowMy(RGB)
title('原始真彩色（256*256*256色）图像')

[X,map] = rgb2ind(RGB,64); % max(RGB(:))=255
imshowMy(X,map)
RGB1 = ind2rgb(X,map); % max(RGB1(:))=0.9961
imshowMy(RGB1)

%% 例 6.1
clc
clear

RGB  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
imshowMy(RGB)
title('原始真彩色（256*256*256色）图像')

[X1, map1] = rgb2ind(RGB, 8, 'nodither');
imview(X1, map1)
% title('颜色数为（8色）无抖动处理后的索引图像')

[X2, map2] = rgb2ind(RGB, 8, 'dither');
imview(X2, map2)
% title('颜色数为（8色）采用抖动处理后的索引图像')

I = rgb2gray(RGB);
imshowMy(I)
title('灰度[亮度]（256色）图像')

I1 = dither(I);
imview(I1)
% title('采用抖动处理后的灰度图像（这是一幅二值图像）')


%% rgb2ntsc
clc
clear
RGB = imread('board.tif');
imshowMy(RGB)

NTSC = rgb2ntsc(RGB);
imshowMy(NTSC)

RGB2 = ntsc2rgb(NTSC);
imshowMy(RGB2)

%% rgb2ycbcr
clc
clear
rgb = imread('board.tif');
imshowMy(rgb)

ycbcr = rgb2ycbcr(rgb);
imshowMy(ycbcr)

rgb2 = ycbcr2rgb(ycbcr);


%% rgb2hsv
clc
clear


%% imcomplement
clc
clear
RGB  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
J = imcomplement(RGB);
imview(RGB), imview(J)


%% rgb2hsi
clc
clear


%% interp1q
clc
clear
z = interp1q([7 255]',[5 255]',[0:275]')

%% spline
clc
clear
x = 0:10;
y = sin(x);
subplot(121),plot(x,y,'o',x,y)
xx = 0:.25:10;
yy = spline(x,y,xx);
subplot(122),plot(x,y,'o',xx,yy)


%% ice
clc
clear
f  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
g = ice('image', f);


%% ice grayscale
clc
clear
info = imfinfo('..\Pictures\images_ch06\Fig0616(a)(Weld Original).tif')
f  = imread('..\Pictures\images_ch06\Fig0616(a)(Weld Original).tif');
g = ice('image', f);

%% 
clc
clear
info = imfinfo('..\Pictures\images_ch06\Fig0616(a)(Weld Original).tif')
f  = imread('..\Pictures\images_ch06\Fig0616(a)(Weld Original).tif');
g = ice('image', f);

%% 
clc
clear

info = imfinfo('..\Pictures\images_ch06\Fig0617(a)(JLK Magenta).tif')
f  = imread('..\Pictures\images_ch06\Fig0617(a)(JLK Magenta).tif');
g = ice('image', f, 'space', 'CMY');

%% 
clc
clear
info = imfinfo('..\Pictures\images_ch06\Fig0618(a)(Caster Original).tif')
f  = imread('..\Pictures\images_ch06\Fig0618(a)(Caster Original).tif');
g = ice('image', f, 'space', 'hsi');

%% 
clc
clear










%% 例6.8 彩色图像平滑 （执行速度较慢）
clc
clear
fc  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
imshowMy(fc)
title('原始真彩色（256*256*256色）图像')

fr = fc(:,:,1);
fg = fc(:,:,2);
fb = fc(:,:,3);

% imshowMy(fr)
% title('红色分量图像')
% imshowMy(fg)
% title('绿色分量图像')
% imshowMy(fb)
% title('蓝色分量图像')

h = rgb2hsi(fc);
H = h(:,:,1);
S = h(:,:,2);
I = h(:,:,3);

% imshowMy(H)
% title('色调分量图像')
% imshowMy(S)
% title('饱和度分量图像')
% imshowMy(I)
% title('亮度分量图像')

w = fspecial('average',15);
I_filtered = imfilter(I,w,'replicate');
h = cat(3,H,S,I_filtered);
f = hsi2rgb(h);
f = min(f,1);
imshowMy(f)
title('仅平滑HSI图像的亮度分量所得到的RGB图像')

fc_filtered = imfilter(fc,w,'replicate');
imshowMy(fc_filtered)
title('分别平滑R、G、B图像分量平面得到的RGB图像')

h_filtered = imfilter(h,w,'replicate');
f = hsi2rgb(h_filtered);
f = min(f,1);
imshowMy(f)
title('分别平滑H、S、I图像分量平面得到的RGB图像')

h_filtered = imfilter(h,w,'replicate');
imshowMy(h_filtered)
title('分别平滑H、S、I图像分量平面得到的HSI图像')

%% 例6.9 彩色图像锐化
clc
clear
fc  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
imshowMy(fc)
title('原始真彩色（256*256*256色）图像')

w = fspecial('average',15);
fc_filtered = imfilter(fc,w,'replicate');
imshowMy(fc_filtered)
title('分别平滑R、G、B图像分量平面得到的RGB模糊图像')

lapmask = [1 1 1; 1 -8 1; 1 1 1];

fen = imsubtract(fc_filtered,imfilter(fc_filtered,lapmask,'replicate'));
imshowMy(fen)
title('用拉普拉斯算子增强模糊图像（效果好象不是很明显！）')

LPA = imfilter(fc,lapmask,'replicate');
imshowMy(LPA)
title('对原始真彩色图像用拉普拉斯算子提取出的图像')

fen = imsubtract(fc,imfilter(fc,lapmask,'replicate'));
imshowMy(fen)
title('用拉普拉斯算子增强原始真彩色图像（采用提高边缘亮度手段）')

%% 例6.10 用函数colorgrad进行RGB边缘检测 图6.24
clc
clear
f  = imread('..\Pictures\images_ch06\Fig0624(d)(RGB2-fullcolor).tif');
imshowMy(f)
title('原始彩色图像')

[VG, A, PPG] = colorgrad(f);

imshowMy(VG) % max(VG(:)) = 1
title('在RGB向量空间计算的梯度（灰度）图像')

imshowMy(PPG) % max(PPG(:)) = 1
title('分别计算R、G、B图像分量平面梯度并将结果相加得到的合成梯度（灰度）图像')

imshowMy(abs(VG - PPG),[])
title('以上两种梯度计算方式的绝对差（灰度）图像（扩展到[ 黑 白 ]）（并不改变数值）')

%% 例6.10 用函数colorgrad进行RGB边缘检测 图6.25
clc
clear
f  = imread('..\Pictures\images_ch06\Fig0619(a)(RGB_iris).tif');
imshowMy(f)
title('原始真彩色（256*256*256色）图像')

[VG, A, PPG] = colorgrad(f);

imshowMy(VG) % max(VG(:)) = 1
title('在RGB向量空间计算的梯度（灰度）图像')

imshowMy(PPG) % max(PPG(:)) = 1
title('分别计算R、G、B图像分量平面梯度并将结果相加得到的合成梯度（灰度）图像')

imshowMy(abs(VG - PPG),[])
title('以上两种梯度计算方式的绝对差（灰度）图像（扩展到[ 黑 白 ]）（并不改变数值）')

%% 例6.11 彩色RGB图像分割 交互选择采样点
clc
clear

f = imread('..\Pictures\images_ch06\Fig0627(a)(jupitermoon_original).tif');
imshowMy(f)
title('原始真彩色（256*256*256色）图像')

figure,mask = roipoly(f);
title('交互选择采样点')
red = immultiply(mask,f(:,:,1));
green = immultiply(mask,f(:,:,2));
blue = immultiply(mask,f(:,:,3));
g = cat(3,red,green,blue);
imshowMy(g)


[M,N,K] = size(g);
I = reshape(g,M*N,3);
idx = find(mask);
I = double(I(idx,1:3));
[C,m] = covmatrix(I);

d = diag(C);
sd = sqrt(d)'

E25 = colorseg('euclidean',f,25,m);
imshowMy(E25)
title('使用圆球体方式描述距离 [ colorseg(euclidean,f,25,m) ] 得到的分割')

E50 = colorseg('euclidean',f,50,m);
imshowMy(E50)
title('使用圆球体方式描述距离 [ colorseg(euclidean,f,50,m) ] 得到的分割')

E75 = colorseg('euclidean',f,75,m);
imshowMy(E75)
title('使用圆球体方式描述距离 [ colorseg(euclidean,f,75,m) ] 得到的分割')

E100 = colorseg('euclidean',f,100,m);
imshowMy(E100)
title('使用圆球体方式描述距离 [ colorseg(euclidean,f,100,m) ] 得到的分割')

M25 = colorseg('mahalanobis',f,25,m,C);
imshowMy(M25)
title('使用椭圆球体方式描述距离 [ colorseg(mahalanobis,f,25,m,C) ] 得到的分割')

M50 = colorseg('mahalanobis',f,50,m,C);
imshowMy(M50)
title('使用椭圆球体方式描述距离 [ colorseg(mahalanobis,f,50,m,C) ] 得到的分割')

M75 = colorseg('mahalanobis',f,75,m,C);
imshowMy(M75)
title('使用椭圆球体方式描述距离 [ colorseg(mahalanobis,f,75,m,C) ] 得到的分割')

M100 = colorseg('mahalanobis',f,100,m,C);
imshowMy(M100)
title('使用椭圆球体方式描述距离 [ colorseg(mahalanobis,f,100,m,C) ] 得到的分割')

%% 
clc
clear


  《临江仙》
      -- 苏轼

夜饮东坡醒复醉，归来仿佛三更。
家僮鼻息已雷鸣。
敲门都不应，倚杖听江声。 

长恨此身非我有，何时忘却营营?
夜阑风静縠纹平。
小舟从此逝，江海寄余生。 






