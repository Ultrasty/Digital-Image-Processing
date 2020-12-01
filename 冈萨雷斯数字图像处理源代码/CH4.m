%% 第四章 频域处理

%% fftshift 对数变换 fftshift 主要用来演示 H 使用
clc
clear
f = imread('..\Pictures\images_ch04\Fig0403(a)(image).tif');
imshowMy(f)
title('原始图像')
imfinfoMy(f)

F = fft2(f);

S = abs(F);
% S(1:5,1:5)
imshowMy(S,[])
title('傅立叶频谱图像')
imfinfoMy(S)

Fc = fftshift(F);
S = abs(Fc);
% S(1:5,1:5)
imshowMy(S)
imshowMy(S,[])
title('居中的傅立叶频谱图像')
imfinfoMy(S)

S = abs(F);
% S(1:5,1:5)
S2 = log(1+S);
% imshowMy(S2)
imshowMy(S2,[])
title('使用对数变换进行视觉增强后的傅立叶频谱图像')
imfinfoMy(S2)

Fc = fftshift(F);
S = abs(Fc);
% S(1:5,1:5)
S2 = log(1+S);
% imshowMy(S2)
imshowMy(S2,[])
title('使用对数变换进行视觉增强并居中后的傅立叶频谱图像')
imfinfoMy(S2)

%% fftshift 对数变换 flower_gray.jpg (速度很慢)
clc
clear
f = imread('..\Pictures\images_ch04\flower_gray.jpg');
imshowMy(f)

f = double(f);

F = fft2(f);



Fc = fftshift(F);
S = abs(Fc);
% S(1:5,1:5)
S2 = log(1+S);
% imshowMy(S2)
imshowMy(S2,[])

%% ifft2 注意！！！imshow(f,[])与imshow(f)的巨大区别！！！
clc
clear
f = imread('..\Pictures\images_ch04\flower_gray.jpg');
imshowMy(f)
imfinfoMy(f)

f = double(f);  % 关键步骤

f(1:8,1:8)
F = fft2(f);

f = real(ifft2(F));
f(1:8,1:8)
imshowMy(f,[])




%% 例4.1 使用填充和不使用填充的滤波效果 lpfilter paddedsize
clc
clear

f = imread('..\Pictures\images_ch04\Fig0405(a)(square_original).tif');
f = im2double(f);
imshowMy(f,[])
imfinfoMy(f)
title('原始图像')

[M, N] = size(f);
F = fft2(f);  % max(F(:)) = 128000
imshowMy(F,[])
imfinfoMy(F)
title('傅立叶频谱（复数）图像')

sig = 10;
H = lpfilter('gaussian',M,N,sig); % max(H(:)) = 1
imshowMy(1-H,[]) % 显示表明滤波器图像未置中
title('滤波器频谱（取反）图像')

G = H.*F;
g = real(ifft2(G));
imshowMy(g,[])
title('不使用填充的频域低通滤波处理后的图像')

PQ = paddedsize(size(f)); % size(f)=[256 256]
Fp = fft2(f, PQ(1),PQ(2));
Hp = lpfilter('gaussian',PQ(1),PQ(2),2*sig); % PQ=[512 512]
imshowMy(fftshift(Hp),[])
Gp = Hp.*Fp;
gp = real(ifft2(Gp));
imshowMy(gp,[])
title('使用填充的频域低通滤波处理后的图像')

gpc = gp(1:size(f,1),1:size(f,2));
imshowMy(gpc,[])
title('使用填充的频域低通滤波处理后的（截取原始大小）图像')
imfinfoMy(gpc)

h = fspecial('gaussian',15,7);
gs = imfilter(f,h);
imshowMy(gs,[])
title('使用空间滤波器处理后的图像')
imfinfoMy(gs)

%% fspecial 实现和上面一段程序同样功能
clc
clear

f = imread('..\Pictures\images_ch04\Fig0405(a)(square_original).tif');

h = fspecial('gaussian',15,7); % 
imshowMy(h,[])

gs = imfilter(f,h);
imshowMy(gs,[])

%% dftfilt P88 一定要注意lpfilter的产生结果 和 使用DFT的 H 先决条件 % H = lpfilter（）% 产生的滤波器原点在左上角
clc
clear
f = imread('..\Pictures\images_ch04\Fig0405(a)(square_original).tif');
f = im2double(f);
imshowMy(f,[])

PQ = paddedsize(size(f)); % size(f)=[256 256]
sig = 10;
H = lpfilter('gaussian',PQ(1),PQ(2),2*sig); % PQ=[512 512]

figure,mesh(abs(H(1:10:512,1:10:512)))

g = dftfilt(f,H);  % 几步合并为一步 % 要求 H 原点在左上角
imshowMy(g,[])

% H1 = ifftshift(H);
% g1 = dftfilt(f,H1);  % 几步合并为一步
% imshowMy(g1,[])

%% 例4.2 空间域滤波和频域滤波的比较 f = im2double(f)
clc
clear
f = imread('..\Pictures\images_ch04\Fig0409(a)(bld).tif');
imfinfoMy(f)
imshowMy(f)
title('原始图像')

F = fft2(f);
S = fftshift(log(1+abs(F)));
S = gscale(S);
imshowMy(S)
title('傅立叶频谱图像')

f = im2double(f); % 转换为
F = fft2(f);
S = fftshift(log(1+abs(F)));
S = gscale(S);
imfinfoMy(S)
imshowMy(S)
title('使用 f = im2double(f) 之后再进行处理的傅立叶频谱图像')
 
%% 例4.2 freqz2 P90 增强垂直边缘 sobel  H = freqz2(h,PQ(1),PQ(2)); % 产生的滤波器原点在矩阵中心处
clc
clear
f = imread('..\Pictures\images_ch04\Fig0409(a)(bld).tif');
imshowMy(f)

F = fft2(f);
S = fftshift(log(1+abs(F)));
S = gscale(S);
imshowMy(S)

h = fspecial('sobel')'; % 增强垂直边缘
%      1     0    -1
%      2     0    -2
%      1     0    -1
figure,freqz2(h); % uses [n2 n1] = [64 64].
% size_h = size(temp)

PQ = paddedsize(size(f));
H = freqz2(h,PQ(1),PQ(2)); % 产生的滤波器原点在矩阵中心处
H1 = ifftshift(H); % 迁移原点到左上角
figure,mesh(abs(H1(1:20:1200,1:20:1200)))

imshowMy(abs(H),[])
imshowMy(abs(H1),[])

gs = imfilter(double(f),h);;
gf = dftfilt(f,H1);

imshowMy(gs,[])
imshowMy(gf,[])

imshowMy(abs(gs),[])
imshowMy(abs(gf),[])

imshowMy(abs(gs) > 0.2*abs(max(gs(:))))
imshowMy(abs(gf) > 0.2*abs(max(gf(:))))


d = abs(gs-gf);
max(d(:))
min(d(:))

%% freqz2 P90 增强水平边缘 sobel % fft2(f) 产生的频域 F 的原点在左上角 
clc
clear
f = imread('..\Pictures\images_ch04\Fig0409(a)(bld).tif');
imshowMy(f)

F = fft2(f);
S = fftshift(log(1+abs(F)));
S = gscale(S);
imshowMy(S)

h = fspecial('sobel'); % 增强水平边缘
figure,freqz2(h);
% size_h = size(temp)

PQ = paddedsize(size(f));
H = freqz2(h,PQ(1),PQ(2));
H1 = ifftshift(H);
figure,mesh(abs(H1(1:20:1200,1:20:1200)))

imshowMy(abs(H),[])
imshowMy(abs(H1),[])

gs = imfilter(double(f),h);;
gf = dftfilt(f,H1);

% imshowMy(gs,[])
% imshowMy(gf,[])
% 
% imshowMy(abs(gs),[])
% imshowMy(abs(gf),[])
% 
% imshowMy(abs(gs) > 0.2*abs(max(gs(:))))
% imshowMy(abs(gf) > 0.2*abs(max(gf(:))))


d = abs(gs-gf);
max(d(:))
min(d(:))

%% freqz2 P90 增强垂直边缘 prewitt
clc
clear
f = imread('..\Pictures\images_ch04\Fig0409(a)(bld).tif');
imshowMy(f)

F = fft2(f);
S = fftshift(log(1+abs(F)));
S = gscale(S);
imshowMy(S)

h = fspecial('prewitt')';
figure,freqz2(h);
% size_h = size(temp)

PQ = paddedsize(size(f));
H = freqz2(h,PQ(1),PQ(2));
H1 = ifftshift(H);
figure,mesh(abs(H1(1:20:1200,1:20:1200)))

imshowMy(abs(H),[])
imshowMy(abs(H1),[])

gs = imfilter(double(f),h);;
gf = dftfilt(f,H1);

imshowMy(gs,[])
imshowMy(gf,[])

imshowMy(abs(gs),[])
imshowMy(abs(gf),[])

imshowMy(abs(gs) > 0.2*abs(max(gs(:))))
imshowMy(abs(gf) > 0.2*abs(max(gf(:))))


d = abs(gs-gf);
max(d(:))
min(d(:))

%% freqz2 P90 增强水平边缘 prewitt
clc
clear
f = imread('..\Pictures\images_ch04\Fig0409(a)(bld).tif');
imshowMy(f)

F = fft2(f);
S = fftshift(log(1+abs(F)));
S = gscale(S);
imshowMy(S)

h = fspecial('prewitt');
figure,freqz2(h);
% size_h = size(temp)

PQ = paddedsize(size(f));
H = freqz2(h,PQ(1),PQ(2));
H1 = ifftshift(H);
figure,mesh(abs(H1(1:20:1200,1:20:1200)))

imshowMy(abs(H),[])
imshowMy(abs(H1),[])

gs = imfilter(double(f),h);;
gf = dftfilt(f,H1);

% imshowMy(gs,[])
% imshowMy(gf,[])
% 
% imshowMy(abs(gs),[])
% imshowMy(abs(gf),[])
% 
% imshowMy(abs(gs) > 0.2*abs(max(gs(:))))
% imshowMy(abs(gf) > 0.2*abs(max(gf(:))))


d = abs(gs-gf);
max(d(:))
min(d(:))


%% 例4.3 dftuv 建立用于实现频域滤波器的网格数组
clc
clear

[U,V] = dftuv(7,5);
% [U,V] = dftuv(8,5);

D = U.^2 + V.^2

fftshift(D)

%% 例4.4 低通滤波器 dftfilt dftuv
clc
clear
f = imread('..\Pictures\images_ch04\Fig0413(a)(original_test_pattern).tif');
imshowMy(f)
title('原始图像')

F1 = fft2(f); % 注意 F1 和 下面 F 的区别
imshowMy(log(1+abs(fftshift(F1))),[])
title('傅立叶频谱图像')

PQ = paddedsize(size(f));
[U V] = dftuv(PQ(1),PQ(2));
D0 = 0.05*PQ(2);

F = fft2(f,PQ(1),PQ(2));
imshowMy(log(1+abs(fftshift(F))),[])
title('傅立叶频谱图像')

H = exp(-(U.^2+V.^2)/(2*(D0^2)));
imshowMy(fftshift(H),[])
title('高斯低通滤波器频谱图像')

g = dftfilt(f,H);
imshowMy(g,[])
title('高斯低通处理后图像')


%% 例4.5 绘制线框图 mesh lpfilter 低通
clc
clear
H = fftshift(lpfilter('gaussian',500,500,50));
mesh(H(1:10:500,1:10:500))
axis([0 50 0 50 0 1])

% colormap([0 0 0])
% axis off
% grid off

imshowMy(H,[])


%% 例4.6 高通滤波器 hpfilter 高通
clc
clear

H = fftshift(hpfilter('ideal',500,500,100)); % 半径是100
% H = fftshift(hpfilter('gaussian',500,500,50));
% H = fftshift(hpfilter('btw',500,500,50));

mesh(H(1:10:500,1:10:500))
axis([0 50 0 50 0 1])

colormap([0 0 0])
axis off
grid off

imshowMy(H,[])

%% 例4.7 高通滤波
clc
clear

f = imread('..\Pictures\images_ch04\Fig0413(a)(original_test_pattern).tif');
imshowMy(f)
title('原始图像')

PQ = paddedsize(size(f));

D0 = 0.05*PQ(1); % 半径是 D0
H = hpfilter('gaussian',PQ(1),PQ(2),D0);
g = dftfilt(f,H);
imshowMy(g,[])
title('高斯高通滤波后的图像')



%% 例4.8 将高频强调滤波和直方图均衡化结合起来 hpfilter histeq
clc
clear
f = imread('..\Pictures\images_ch04\Fig0419(a)(chestXray_original).tif');
imshowMy(f)
title('原始图像')
imfinfoMy(f)

PQ = paddedsize(size(f));
D0 = 0.05*PQ(1);
HBW = hpfilter('btw',PQ(1),PQ(2),D0,2);
H = 0.5+2*HBW;
gbw = dftfilt(f,HBW);
% 使用了 gscale(gbw) 之后，imshowMy(gbw) 等价于 imshowMy(gbw,[])
gbw = gscale(gbw); 
imshowMy(gbw,[])
title('高通滤波后的图像')
imfinfoMy(gbw)

gbe = histeq(gbw,256);
imshowMy(gbe,[])
title('高通滤波并经过直方图均衡化后的图像')
imfinfoMy(gbe)

ghf = dftfilt(f,H);
ghf = gscale(ghf);
imshowMy(ghf,[])
title('高频强调滤波后的图像')
imfinfoMy(ghf)

ghe = histeq(ghf,256);
imshowMy(ghe,[])
title('高频强调滤波并经过直方图均衡化后的图像')
imfinfoMy(ghe)

%% fftshift ifftshift 
clc
clear

A = [2 0 0 1
     0 0 0 0
     0 0 0 0
     3 0 0 4]

fftshift(A)

fftshift(fftshift(A))

ifftshift(fftshift(A))



%% 注意!!!
freqz2 生成的滤波器原点在正中央
lpfilter（低通）生成的滤波器原点在左上角
hpfilter（高通）生成的滤波器原点在左上角

%% 
clc
clear


《念奴娇·过洞庭》
      -- 张孝祥 
      
洞庭青草，近中秋、更无一点风色。 
玉鉴琼田三万顷，著我扁舟一叶。 
素月分辉，明河共影，表里俱澄澈。 
悠然心会，妙处难与君说。 

应念岭表经年，孤光自照，肝胆皆冰雪。 
短发萧骚襟袖冷，稳泛沧溟空阔。 
尽挹西江，细斟北斗，万象为宾客。 
扣舷独啸，不知今夕何夕。 

