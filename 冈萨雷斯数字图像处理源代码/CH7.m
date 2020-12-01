%% 第七章 小波

%% 例7.1 wfilters wavefun waveinfo 运行速度很慢！
clc
clear

% Set wavelet name. 
% wname = 'sym2';
wname = 'haar';

% Compute the four filters associated with wavelet name given 
% by the input string wname. 
[Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(wname);
subplot(221); stem(Lo_D); 
title('Decomposition low-pass filter'); 
subplot(222); stem(Hi_D); 
title('Decomposition high-pass filter'); 
subplot(223); stem(Lo_R); 
title('Reconstruction low-pass filter'); 
subplot(224); stem(Hi_R); 
title('Reconstruction high-pass filter'); 
xlabel('The four filters for db5')

% Editing some graphical properties,
% the following figure is generated.

[phi,psi,xval] = wavefun(wname,10);
xaxis = zeros(size(xval));
figure,subplot(121);plot(xval,phi,'k',xval,xaxis,'--k');
% axis([-0.02 1.01 -1.5 1.5]);axis square;
title('Scaling Function');
subplot(122);plot(xval,psi,'k',xval,xaxis,'--k');
% axis([-0.02 1.01 -1.5 1.5]);axis square;
title('Wavelet Function');


%% wavefun
clc
clear

% [PHI1,PSI1,PHI2,PSI2,XVAL] = wavefun('haar');
[PHI1,PSI1,XVAL] = wavefun('haar');

waveinfo('haar')

%% 例7.2 wavedec2
clc
clear

f = magic(8)

[c0,s0] = wavedec2(f,0,'haar')

[c1,s1] = wavedec2(f,1,'haar')

[c2,s2] = wavedec2(f,2,'haar')

[c3,s3] = wavedec2(f,3,'haar')

[c4,s4] = wavedec2(f,4,'haar')

[c10,s10] = wavedec2(f,10,'haar');

%% 例7.3 wavefast fwtcompare
clc
clear

f = imread('..\Pictures\images_ch07\Fig0704(Vase).tif');

[ratio maxdifference] = fwtcompare(f,6,'db4')




%% 例7.4 wavedec2  appcoef2 detcoef2 阈值化处理
clc
clear

f = magic(8)

[c3,s3] = wavedec2(f,3,'haar')

approx = appcoef2(c3,s3,'haar')
approx0 = appcoef2(c3,s3,'haar',0)
approx1 = appcoef2(c3,s3,'haar',1)
approx2 = appcoef2(c3,s3,'haar',2)
approx3 = appcoef2(c3,s3,'haar',3)

horizdet3 = detcoef2('h',c3,s3,3)
horizdet2 = detcoef2('h',c3,s3,2)
horizdet1 = detcoef2('h',c3,s3,1)
% horizdet0 = detcoef2('h',c3,s3,0) % 错误的，细节没有是0阶的

newc3 = wthcoef2('h',c3,s3,2);
newhorizdet2 = detcoef2('h',newc3,s3,2)

% 硬阈值
newc3 = wthcoef2('d',c3,s3,1,46,'h'); % 阈值：46
newdiagon1_hard = detcoef2('d',newc3,s3,1)
% 软阈值
newc3 = wthcoef2('d',c3,s3,1,46,'s'); % 阈值：46
newdiagon1_soft = detcoef2('d',newc3,s3,1)

%% 例7.5 wavecut wavecopy(使用本书自带的函数)
clc
clear

f = magic(8)

[c3,s3] = wavedec2(f,3,'haar')

approx = wavecopy('a',c3,s3)
approx1 = wavecopy('a',c3,s3,1) % 没有重构，只能提取最后的近似（不如appcoef2强大）

horizdet = wavecopy('h',c3,s3) % 默认时则提取第一级细节系数
horizdet3 = wavecopy('h',c3,s3,3)
horizdet2 = wavecopy('h',c3,s3,2)

[newc3,horizdet2] = wavecut('h',c3,s3,2)
newhorizdet2 = wavecopy('h',newc3,s3,2)


%% 例7.6 一层分解 子带
clc
clear
f = imread('..\Pictures\images_ch07\Fig0704(Vase).tif');
[c,s] = wavefast(f,1,'db4'); % 一层分解
wave2gray(c,s);
title('[db4]自动缩放')

figure,wave2gray(c,s,8);
title('[db4]细节系数8倍放大')

figure,wave2gray(c,s,-8);
title('[db4]细节系数8倍放大的绝对值')


%% 例7.6  二层分解 子带
clc
clear
f = imread('..\Pictures\images_ch07\Fig0704(Vase).tif');
[c,s] = wavefast(f,2,'db4'); % 二层分解
s

wave2gray(c,s);
title('[db4]自动缩放')

figure,wave2gray(c,s,8);
title('[db4]细节系数8倍放大')

figure,wave2gray(c,s,-8);
title('[db4]细节系数8倍放大的绝对值')

%% 例7.6  二层分解 子带 haar
clc
clear
f = imread('..\Pictures\images_ch07\Fig0704(Vase).tif');
[c,s] = wavefast(f,2,'haar'); % 二层分解
wave2gray(c,s);
title('[haar]自动缩放')

figure,wave2gray(c,s,8);
title('[haar]细节系数8倍放大')

figure,wave2gray(c,s,8,'append');
title('[haar]细节系数8倍放大（使用 append 参数）')

figure,wave2gray(c,s,-8);
title('[haar]细节系数8倍放大的绝对值')

%% 例7.7 重建图像 waveback waverec2
clc
clear

f = imread('..\Pictures\images_ch07\Fig0704(Vase).tif');
imshowMy(f)

[c,s] = wavefast(f,2,'haar'); % 二层分解
figure,wave2gray(c,s);
title('[haar]自动缩放')

g = waveback(c,s,'haar');
g = uint8(g);
imshowMy(g)

g1 = waverec2(c,s,'haar');
g1 = uint8(g1);
imshowMy(g1)

%% 例7.8 小波的方向性和边缘检测 P207
clc
clear
f = imread('..\Pictures\images_ch07\Fig0707(a)(Original).tif');
imshow(f);
title('原始测试图像')

[c s] = wavefast(f,1,'sym4');
figure,wave2gray(c,s,-6);
title('[sym4]小波变换结果')

[nc,y] = wavecut('a',c,s);
figure,wave2gray(nc,s,-6);
title('将所有近似系数（低频部分）设置为0的变换')

edges = abs(waveback(nc,s,'sym4'));
figure;imshow(mat2gray(edges));
title('计算反变换的绝对值所得到的边缘（高频细节）图像')

%% 例7.8 小波的方向性和边缘检测
clc
clear
f = imread('..\Pictures\images_ch07\Fig0707(a)(Original).tif');
imshow(f);
[c s] = wavefast(f,1,'sym4');
figure,wave2gray(c,s);

[nc,y] = wavecut('a',c,s);
figure,wave2gray(nc,s);

edges = abs(waveback(nc,s,'sym4'));
figure;imshow(mat2gray(edges));

%% 例7.9 基于小波的图像平滑或模糊
clc
clear
f = imread('..\Pictures\images_ch07\Fig0707(a)(Original).tif');
imshow(f);
title('原始测试图像')

[c s] = wavefast(f,4,'sym4');
figure,wave2gray(c,s,20);
title('[sym4]小波变换结果（四次分解）')

[c,g8] = wavezero(c,s,1,'sym4');
title('仅将第一级的细节系数设置为0后的反变换')

[c,g8] = wavezero(c,s,2,'sym4');
title('将第一、二级的细节系数设置为0后的反变换')

[c,g8] = wavezero(c,s,3,'sym4');
title('将第一、二、三级的细节系数设置为0后的反变换')

[c,g8] = wavezero(c,s,4,'sym4');
title('将第一、二、三、四级的细节系数设置为0后的反变换')


%% 例7.9 基于小波的图像平滑或模糊
clc
clear
f = imread('..\Pictures\images_ch07\Fig0707(a)(Original).tif');
imshow(f);
title('原始测试图像')

[c0 s] = wavefast(f,4,'sym4');
figure,wave2gray(c0,s,20);
title('[sym4]小波变换结果（四次分解）')

[c,g8] = wavezero(c0,s,1,'sym4');
title('仅将第一级的细节系数设置为0后的反变换（去掉人眼不易觉察出来的高频细节）')

[c,g8] = wavezero(c0,s,2,'sym4');
title('仅将第二级的细节系数设置为0后的反变换')

[c,g8] = wavezero(c0,s,3,'sym4');
title('仅将第三级的细节系数设置为0后的反变换')

[c,g8] = wavezero(c0,s,4,'sym4');
title('仅将第四级的细节系数设置为0后的反变换')





%% 例7.10 渐进重构
clc
clear
f = imread('..\Pictures\images_ch07\Fig0709(original_strawberries).tif');
imshow(f)

[c s] = wavefast(f,4,'jpeg9.7');
figure,wave2gray(c,s,8);

f = wavecopy('a',c,s);
figure;imshow(mat2gray(f));

[c s] = waveback(c,s,'jpeg9.7',1);
f = wavecopy('a',c,s);
figure;imshow(mat2gray(f));

[c s] = waveback(c,s,'jpeg9.7',1);
f = wavecopy('a',c,s);
figure;imshow(mat2gray(f));

[c s] = waveback(c,s,'jpeg9.7',1);
f = wavecopy('a',c,s);
figure;imshow(mat2gray(f));

[c s] = waveback(c,s,'jpeg9.7',1);
f = wavecopy('a',c,s);
figure;imshow(mat2gray(f));

%% 例7.10 渐进重构
clc
clear
f = imread('..\Pictures\images_ch07\Fig0704(Vase).tif');
imshow(f)

[c s] = wavefast(f,4,'jpeg9.7');
figure,wave2gray(c,s,8);

% f = wavecopy('a',c,s);
% figure;imshow(mat2gray(f));

[c s] = waveback(c,s,'jpeg9.7',4);
f = wavecopy('a',c,s);
figure;imshow(mat2gray(f));

%% 
clc
clear

《金缕曲·季子平安否》
    --- 顾贞观

季子平安否 
便归来 平生万事 那堪回首 
行路悠悠谁慰藉 母老家贫子幼 
记不起 从前杯酒 
魑魅搏人应见惯 总输他翻云覆雨手 
冰与雪 周旋久 

泪痕莫滴牛衣透 
数天涯 依然骨肉 几家能够 
比似红颜多命薄 更不如今还有 
只绝塞 苦寒难受 
廿载包胥成一诺 盼乌头马角终相救 
置此札 君怀袖 

我亦飘零久 
十年来 深恩负尽 死生师友 
宿昔齐名非忝窃 试看杜陵消瘦 
曾不减 夜郎潺愁 
薄命长辞知己别 问人生到此凄凉否 
千万恨 为君剖 

兄生辛未吾丁丑 
共些时 冰霜摧折 早衰蒲柳 
诗赋从今须少作 留取心魂相守 
但愿得河清人寿 
归日急翻行戍稿 把空名料理传身后 
言不尽 观顿首 







