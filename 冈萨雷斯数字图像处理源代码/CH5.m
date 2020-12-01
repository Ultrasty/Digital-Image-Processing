%% 第五章 图像复原

%% 例5.2 各种噪声
clc
clear

r = imnoise2('gaussian',100000,1,0,1);
bins = 100;
hist(r,bins)
title('gaussian')

r = imnoise2('uniform',100000,1,0,1);
bins = 100;
figure,hist(r,bins)
title('uniform')

r = imnoise2('salt & pepper',1000,1,0.1,0.27);
bins = 100;
figure,hist(r,bins)
title('salt & pepper')

r = imnoise2('lognormal',100000,1);
bins = 100;
figure,hist(r,bins)
title('lognormal')

r = imnoise2('rayleigh',100000,1,0,1);
bins = 100;
figure,hist(r,bins)
title('rayleigh')

r = imnoise2('exponential',100000,1);
bins = 100;
figure,hist(r,bins)
title('exponential')

r = imnoise2('erlang',100000,1);
bins = 100;
figure,hist(r,bins)
title('erlang')




%% 例5.3 imnoise3 
clc
clear
C = [0 64; 0 128; 32 32; 64 0; 128 0; -32 32];
[r,R,S] = imnoise3(512, 512, C);
imshowMy(S,[])
imfinfoMy(S)
title('[6个]指定冲击的正弦噪声周期频谱[1]')

imshowMy(abs(R),[])

imshowMy(r,[])
imfinfoMy(r)
title('[6个]相应的正弦噪声周期模式[1]')
% S1 = fftshift(S);
% imshowMy(S1,[])
% figure,mesh(S)


C1 = C/2;
[r,R,S] = imnoise3(512, 512, C1);
imshowMy(S,[])
title('[6个]指定冲击的正弦噪声周期频谱[2]')
imshowMy(r,[])
title('[6个]相应的正弦噪声周期模式[2]')

C2 = [6 32; -2 2];
[r,R,S] = imnoise3(512, 512, C2);
imshowMy(S,[])
title('[2个]指定冲击的正弦噪声周期频谱[3]')
imshowMy(r,[])
title('[2个]相应的正弦噪声周期模式[3]')

A = [1 5];
[r,R,S] = imnoise3(512, 512, C2, A);
imshowMy(1-S,[])  %有两个不清楚的点，因为其振幅较小
title('[2个][使用非默认的不同振幅]指定冲击的正弦噪声周期频谱[4]')
imshowMy(r,[])
title('[2个][使用非默认的不同振幅]相应的正弦噪声周期模式[4]')


%% imnoise3 
clc
clear
C1 = [6 32];
[r,R,S] = imnoise3(512, 512, C1);
imshowMy(S,[])
title('[1个]指定冲击的正弦噪声周期频谱[1]')
imshowMy(r,[])
title('[1个]相应的正弦噪声周期模式[1]')

C2 = [ -2 2];
[r,R,S] = imnoise3(512, 512, C2);
imshowMy(S,[])
title('[1个]指定冲击的正弦噪声周期频谱[2]')
imshowMy(r,[])
title('[1个]相应的正弦噪声周期模式[2]')


C3 = [6 32; -2 2];
A = [1 5];
[r,R,S] = imnoise3(512, 512, C3, A);
imshowMy(1-S,[])  %有两个不清楚的点，因为其振幅较小
title('[2个][使用非默认的不同振幅]指定冲击的正弦噪声周期频谱[1]')
imshowMy(r,[])
title('[2个][使用非默认的不同振幅]相应的正弦噪声周期模式[1]')


C3 = [6 32; -2 2];
A = [5 1];
[r,R,S] = imnoise3(512, 512, C3, A);
imshowMy(1-S,[])  %有两个不清楚的点，因为其振幅较小
title('[2个][使用非默认的不同振幅]指定冲击的正弦噪声周期频谱[2]')
imshowMy(r,[])
title('[2个][使用非默认的不同振幅]相应的正弦噪声周期模式[2]')


%% 例5.4 估计噪声参数    交互式选取区域产生的直方图
clc
clear

f = imread('..\Pictures\images_ch05\Fig0504(a)(noisy_image).tif');
imshow(f)
title('原始含噪声图像')

[B,c,r] = roipoly(f);
figure,imshow(B)

[p,npix] = histroi(f,c,r);
figure,bar(p,1)
title('交互式选取区域产生的直方图')
axis tight

[v,unv] = statmoments(p,2) % ????

X = imnoise2('gaussian',npix,1, unv(1), sqrt(unv(2)) );
figure,hist(X,130)
title('使用函数[imnoise2]产生的高斯数据的直方图')
% axis([0 300 0 140])
axis tight

%% 掩模的使用方法 P114
clc
clear

f = imread('..\Pictures\images_ch05\Fig0504(a)(noisy_image).tif');
imshow(f)

[B,c,r] = roipoly(f);
roi = f(B);

size_f = size(f)
class_f = class(f)
size_B = size(B)
class_B = class(B)
size_roi = size(roi) % 列向量


%% 例5.5 spfilt 空间噪声滤波器
clc
clear
f = imread('..\Pictures\images_ch05\Fig0318(a)(ckt-board-orig).tif');
imshowMy(f)
title('原始图像')

[M,N] = size(f);
R = imnoise2('salt & pepper',M,N,0.1,0);
c = find(R == 0);
gp = f;
gp(c) = 0;
imshowMy(gp)
title('被概率为0.1的胡椒噪声污染的图像')

R = imnoise2('salt & pepper',M,N,0,0.1);
c = find(R == 1);
gs = f;
gs(c) = 255;
imshowMy(gs)
title('被概率为0.1的盐粒噪声污染的图像')

fp = spfilt(gp,'chmean',3,3,1.5);
imshowMy(fp)
title('用阶为Q=1.5的3*3反调和滤波器对[被概率为0.1的胡椒噪声污染的图像]滤波的结果')

fs = spfilt(gs,'chmean',3,3,-1.5);
imshowMy(fs)
title('用阶为Q=-1.5的3*3反调和滤波器对[被概率为0.1的盐粒噪声污染的图像]滤波的结果')

fpmax = spfilt(gp,'max',3,3);
imshowMy(fpmax)
title('用3*3最大滤波器对[被概率为0.1的胡椒噪声污染的图像]滤波的结果')

fsmin = spfilt(gs,'min',3,3);
imshowMy(fsmin)
title('用3*3最小滤波器对[被概率为0.1的盐粒噪声污染的图像]滤波的结果')




%% 例5.6 自适应中值滤波 adpmedian
clc
clear
f = imread('..\Pictures\images_ch05\Fig0318(a)(ckt-board-orig).tif');
imshowMy(f)
title('原始图像')

g = imnoise(f,'salt & pepper',0.25);% 噪声点有黑有白
imshowMy(g)
title('被概率为0.25椒盐噪声污染的图像')

f1 = medfilt2(g,[7 7],'symmetric');
imshowMy(f1)
title('用7*7中值滤波器对[被概率为0.25椒盐噪声污染的图像]滤波的结果')

f2 = adpmedian(g,7);
imshowMy(f2)
title('用Smax=7的自适应中值滤波器对[被概率为0.25椒盐噪声污染的图像]滤波的结果')

%% 例5.7 模糊噪声图像建模 fspecial imfilter pixeldup
clc
clear

f = checkerboard(8);
imshowMy(f)
title('原始图像')

PSF = fspecial('motion',7,45);  % sum(PSF(:)) = 1
gb = imfilter(f,PSF,'circular');
imshowMy(gb)
title('使用 PSF = fspecial(motion,7,45) 模糊后的图像')

noise = imnoise(zeros(size(f)),'gaussian',0,0.001);
imshowMy(noise,[])
title('高斯纯噪声图像')

g = gb + noise;
imshowMy(g,[])
title('模糊加噪声的图像')

% imshowMy(pixeldup(f,8),[])


%% 例5.8 使用 deconvwnr 函数复原模糊噪声图像
clc
clear

f = checkerboard(8);
% imshowMy(f)

PSF = fspecial('motion',7,45)
gb = imfilter(f,PSF,'circular');
% imshowMy(gb)

noise = imnoise(zeros(size(f)),'gaussian',0,0.001);
% imshowMy(noise,[])

g = gb + noise;
imshowMy(g,[])
title('模糊加噪声的图像')
% ***************

fr1 = deconvwnr(g,PSF);
imshowMy(fr1,[])
title('简单的维纳滤波（逆滤波）后的结果')

Sn = abs(fft2(noise)).^2;
nA = sum(Sn(:))/prod(size(noise));
Sf = abs(fft2(f)).^2;
fA = sum(Sf(:))/prod(size(f));
R = nA/fA;

fr2 = deconvwnr(g,PSF,R);
imshowMy(fr2,[])
title('使用常数比率的维纳滤波后的结果')

NCORR = fftshift(real(ifft(Sn)));
ICORR = fftshift(real(ifft(Sf)));
fr3 = deconvwnr(g,PSF,NCORR,ICORR);
imshowMy(fr3,[])
title('使用自相关函数的维纳滤波后的结果')

% imshowMy(pixeldup(fr3,8))
%% 例5.8  当模糊但不含噪声时使用维纳滤波还原
clc
clear

f = checkerboard(8);
imshowMy(f)

PSF = fspecial('motion',7,45)
gb = imfilter(f,PSF,'circular');
imshowMy(gb)

% ***************
g = gb;

fr1 = deconvwnr(g,PSF);
imshowMy(fr1,[])

% Sn = abs(fft2(noise)).^2;
% nA = sum(Sn(:))/prod(size(noise));
% Sf = abs(fft2(f)).^2;
% fA = sum(Sf(:))/prod(size(f));
% R = nA/fA;
% 
% fr2 = deconvwnr(g,PSF,R);
% imshowMy(fr2,[])
% 
% NCORR = fftshift(real(ifft(Sn)));
% ICORR = fftshift(real(ifft(Sf)));
% fr3 = deconvwnr(g,PSF,NCORR,ICORR);
% imshowMy(fr3,[])


%% 例5.9 约束的最小二乘方（正则）滤波deconvreg
clc
clear
f = checkerboard(8);
% imshowMy(f)

PSF = fspecial('motion',7,45)
gb = imfilter(f,PSF,'circular');
% imshowMy(gb)

noise = imnoise(zeros(size(f)),'gaussian',0,0.001);
% imshowMy(noise,[])

g = gb + noise;
imshowMy(g,[])
title('模糊加噪声的图像')
% **************


fr1 = deconvreg(g,PSF,4);
imshowMy(fr1,[])
title('使用 fr1 = deconvreg(g,PSF,4) 正则滤波后的结果')

fr2 = deconvreg(g,PSF,0.4,[1e-7 1e7]);
imshowMy(fr2,[])
title('使用 fr2 = deconvreg(g,PSF,0.4,[1e-7 1e7]) 正则滤波后的结果')

% imshowMy(pixeldup(fr2,8))
%% 例5.10 使用L-R 算法的迭代非线性复原 deconvlucy （这是一种非线性复原算法）
clc
clear

f = checkerboard(8);
% imshow(f)
imshowMy(pixeldup(f,8))
title('原始图像')

PSF = fspecial('gaussian',7,10)

SD = 0.01;
g = imnoise(imfilter(f,PSF),'gaussian',0,SD^2);
imshowMy(pixeldup(g,8))
title('模糊加噪声的图像')

DAMPAR = 10*SD;

LIM = ceil(size(PSF,1)/2);
WEIGHT = zeros(size(g));
WEIGHT(LIM + 1:end - LIM, LIM + 1:end - LIM) = 1;

% ------------

NUMIT = 5;
f5 = deconvlucy(g,PSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(f5,8))
title('使用L-R 算法的迭代次数为5的非线性复原后的图像')

NUMIT = 10;
f10 = deconvlucy(g,PSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(f10,8))
title('使用L-R 算法的迭代次数为10的非线性复原后的图像')

NUMIT = 20;
f20 = deconvlucy(g,PSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(f20,8))
title('使用L-R 算法的迭代次数为20的非线性复原后的图像')

NUMIT = 100;
f100 = deconvlucy(g,PSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(f100,8))
title('使用L-R 算法的迭代次数为100的非线性复原后的图像')

NUMIT = 1000;
f1000 = deconvlucy(g,PSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(f1000,8))
title('使用L-R 算法的迭代次数为1000的非线性复原后的图像')

%% 例5.11 盲去卷积 估计PSF deconvblind
clc
clear

PSF = fspecial('gaussian',7,10)
imshowMy(pixeldup(PSF,73),[])
title('原始PSF图像')

f = checkerboard(8);
SD = 0.01;
g = imnoise(imfilter(f,PSF),'gaussian',0,SD^2);

INITPSF = ones(size(PSF)); % 初始值(尺寸大小与原始PSF一样)

DAMPAR = 10*SD;
LIM = ceil(size(PSF,1)/2);
WEIGHT = zeros(size(g));
WEIGHT(LIM + 1:end - LIM, LIM + 1:end - LIM) = 1;

NUMIT = 5;
[fr PSFe] = deconvblind(g,INITPSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(PSFe,73),[])
title('使用盲去卷积估计PSF迭代5次后的结果')

NUMIT = 10;
[fr PSFe] = deconvblind(g,INITPSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(PSFe,73),[])
title('使用盲去卷积估计PSF迭代10次后的结果')

NUMIT = 20;
[fr PSFe] = deconvblind(g,INITPSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(PSFe,73),[])
title('使用盲去卷积估计PSF迭代20次后的结果')

NUMIT = 50; % 并非迭代次数越多越好！！！
[fr PSFe] = deconvblind(g,INITPSF,NUMIT,DAMPAR,WEIGHT);
imshowMy(pixeldup(PSFe,73),[])
title('使用盲去卷积估计PSF迭代50次后的结果')

%% 例5.12  vistformfwd
clc
clear

T1 = [3 0 0; 0 2 0; 0 0 1];
tform1 = maketform('affine',T1);
vistformfwd(tform1,[0 100],[0 100]);

T2 = [1 0 0; 0.2 1 0; 0 0 1];
tform2 = maketform('affine',T2);
figure,vistformfwd(tform2,[0 100],[0 100]);

Tscale = [1.5 0 0; 0 2 0; 0 0 1];
Trotation = [cos(pi/4) sin(pi/4) 0
            -sin(pi/4) cos(pi/4) 0
            0 0 1];
Tshear = [1 0 0; 0.2 1 0; 0 0 1];
T3 = Tscale * Trotation * Tshear;
tform3 = maketform('affine',T3);
figure,vistformfwd(tform3,[0 100],[0 100]);
        
            
%% 例5.12  vistformfwd
clc
clear

Tscale = [1.5 0 0; 0 2 0; 0 0 1];
Trotation = [cos(pi/4) sin(pi/4) 0
            -sin(pi/4) cos(pi/4) 0
            0 0 1];

T1 = Tscale * Trotation;
tform1 = maketform('affine',T1);
figure,vistformfwd(tform1,[0 100],[0 100]);

Tscale = [1.5 0 0; 0 2 0; 0 0 1];
Trotation = [cos(pi/4) sin(pi/4) 0
            -sin(pi/4) cos(pi/4) 0
            0 0 1];
Tshear = [1 0 0; 0.2 1 0; 0 0 1];
T3 = Tscale * Trotation * Tshear;
tform3 = maketform('affine',T3);
figure,vistformfwd(tform3,[0 100],[0 100]);
%% 例5.13 图像空间变换
clc
clear

f = checkerboard(50);
imshowMy(f,[])

s = 1;
theta = pi/6;
T = [s*cos(theta) s*sin(theta) 0
            -s*sin(theta) s*cos(theta) 0
            0 0 1];
tform = maketform('affine',T);
g = imtransform(f,tform);
imshowMy(g,[])

g2 = imtransform(f,tform,'nearest');
imshowMy(g2,[])

g3 = imtransform(f,tform,'FillValue',0.5);
imshowMy(g3,[])

T2 = [1 0 0; 0 1 0; 50 50 1];
tform2 = maketform('affine',T2);
g4 = imtransform(f,tform2);
% imshowMy(g4,[])

g5 = imtransform(f,tform2,'XData',[1 500],'YData',[1 500],...
                 'FillValue',0.5);
imshowMy(g5,[])



%% cpselect ??? cp2tform
clc
clear

g = imread('..\Pictures\images_ch05\Fig0515(a)(base-with-control-points).tif');
imshowMy(g)

basepoints = [83 81; 450 56 ; 43 293; 249 392; 436 442];
inputpoints = [68 66; 375 47 ; 42 286 ;275 434; 523 532];

tform = cp2tform(inputpoints, basepoints, 'projective');
gp = imtransform(g,tform,'XData', [1 502],'YData',[1 502]);

imshowMy(gp)




%% 本章一些经验总结

如果我们已经对噪声和图像谱的知识有足够的了解的前提下。Wiener 滤波结果要好得多。
如果没有这些信息，则用“约束的最小二乘方（正则）”滤波器和 Wiener滤波 基本差不多效果。（P130）



%% 
clc
clear

闻王昌龄左迁龙标遥有此寄 
      --- 李白 

杨花落尽子规啼，闻道龙标过五溪[1]。 
我寄愁心与明月，随风直到夜郎西[2]。 




