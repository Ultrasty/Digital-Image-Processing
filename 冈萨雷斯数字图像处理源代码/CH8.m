%% 第八章 图像压缩

%% 衡量压缩前后的误差 compare P213
clc
clear
f1 = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy).tif');
imshowMy(f1)
imfinfoMy(f1)

f12 = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy).jpg'); % 品质为12时
imshowMy(f12)
imfinfoMy(f12)

f05 = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy05).jpg'); % 品质为5时
imshowMy(f05)
imfinfoMy(f05)

rmes12 = compare(f1,f12)

rmes05 = compare(f1,f05,15)

%% 例8.1 entropy 熵
clc
clear

f = [119 123 168 119; 123 119 168 168];
f = [f;119 119 107 119; 107 107 119 119]
[p x] = hist(f(:),8)
hist(f(:),8)
p = p/sum(p)
h = entropy(f)
c = huffman(hist(double(f(:)),4))

cp = huffman([0.1875 0.5 0.125 0.1875])

%% 例8.2 dec2bin huffman mat2huff P222
clc
clear

% f2 = uint8([2 3 4.3 2; 3 2 9 4; 2 2 1 2; 1 1 2 2])
f2 = [2 3 4.3 2; 3 2 9.8 4; 2 2 1 2; 1 1 2 2]
R1 = whos('f2')

c = huffman(hist(double(f2(:)),5))

% h1f2 = c(f2(:))'
% whos('h1f2')
% 
% h2f2 = char(h1f2)'
% whos('h2f2')
% 
% % 1010011000011011
% %  1 11  1001  0  
% %  0 10  1  1     
% 
% h2f2 = h2f2(:);
% h2f2(h2f2 == ' ') = [];
% h2f2
% whos('h2f2')

h3f2 = mat2huff(f2) % 编码
whos('h3f2')

g = huff2mat(h3f2) % 解码 注意解码并不完美 4.3被解释为4, 9.8被解释为10

hcode = h3f2.code;
R2 = whos('hcode')

dec2bin(double(hcode))

ratio = R1.bytes/R2.bytes % 粗糙的压缩比率
% dec2bin(9)

%% cell ???
clc
clear
X = cell(2, 3)

X{1}= {8,9}
X{1}
X(1)

X{2} = 5 % 等价于 X{2} = {5}or[5]
X{2}
X(2)

X(3) = {6} % 等价于 X[3] = {[6]} 注意：6or[6]均是错误的
X{3}
X(3)


X(4) = {[7 9]}
X{4}
X(4)

X{5} = {[10,11]}
X{5}
X(5)

X{6} = [12,13]
X{6}
X(6)

celldisp(X)
cellplot(X)


%% 剪去第一个元素的实现方法
clc
clear
p = [1 2 3 4]
p(1) = []
p

%% 例8.2 变长码映射
clc
clear

f2 = uint8([236 3 4 2; 3 2 4 4; 2 2 100 2; 1 1 2 2])
whos('f2')

c = huffman(hist(double(f2(:)),6))


h3f2 = mat2huff(f2)
whos('h3f2')

% **************************************
hcode = h3f2.code
whos('hcode')

% 结果解释
% h3f2 = 
% 
%     size: [4 4]                矩阵大小形态
%      min: 32769                矩阵最小值
%     hist: [1x236 uint16]       最小值到最大值之间的分布
%     code: [5419 11655 30720]   对矩阵从上到下、从左到右编码得到的结果

% **************************************

dec2bin(double(hcode))




%% 例8.3 使用mat2huff进行编码
clc
clear

f = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy).tif');
c = mat2huff(f);
cr1 = imratio(f,c)

save ..\Data\SqueezeTracy c;
cr2 = imratio('..\Pictures\images_ch08\Fig0804(a)(Tracy).tif','..\Data\SqueezeTracy.mat')


%% 例8.4 使用huff2mat解码
clc
clear
load ..\Data\SqueezeTracy.mat
g = huff2mat(c);
f = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy).tif');
rmse = compare(f,g)




%% 例8.5 无损预测编码
clc
clear

f = imread('..\Pictures\images_ch08\Fig0807(c)(Aligned).tif');
imshowMy(f)

fShang = entropy(f)
imfinfoMy(f)
c0 = mat2huff(double(f));
cfShang = entropy(c0.code)
cfr = imratio(f,c0)

% *************** 编码 ********
e = mat2lpc(f);
imfinfoMy(e)
imshowMy(mat2gray(e));
eShang = entropy(e)
cer = imratio(f,uint8(e))

c = mat2huff(e);
ceShang = entropy(c.code)
cr = imratio(f,c)
% ****************************

% *************** 解码 ********
ee = huff2mat(c);
ff = lpc2mat(ee);
imshowMy(ff,[])
% ****************************

[h,x] = hist(e(:)*512,512);
figure,bar(x,h,'k')
set(gcf,'outerposition',get(0,'screensize'))
title('预测误差的直方图')

g = lpc2mat(huff2mat(c));
compare(f,g) % 计算前后图像像素误差

%% 例8.6
clc
clear






%% 例8.7 利用无损预测和霍夫曼编码的混合IGS量化（16）
clc
clear
f = imread('..\Pictures\images_ch08\Fig0810(a)(Original).tif');
imshowMy(f)
title('原始图像')
imfinfoMy(f)

imshowMy(f, 16)
title('均匀量化为16级灰度图像')

% ********************************
% [Y, newmap] = imapprox(f,16);
% imshowMy(Y, newmap)
% title('索引图像（16色）')
% 
% % Y = grayslice(mat2gray(double(f)),16);
% % imshowMy(Y)
% % title('索引图像（16色）')
% 
% Y= ind2gray(f,16);
% imshowMy(Y, newmap)
% title('索引图像（16色）')
% ********************************

q = quantize(f,4,'igs');
imshowMy(q,[])
imfinfoMy(q)
title('(改进的灰度级量化)IGS量化为16级灰度图像')

qs = double(q)/16;
imshowMy(qs,[])
imfinfoMy(qs)
title('IGS量化为16级灰度图像')

e = mat2lpc(qs);
c = mat2huff(e);
cr = imratio(f,c)
% ------------------
ne = huff2mat(c);
nqs = lpc2mat(ne);
% imshowMy(nqs)
nq = 16*nqs;
% imshowMy(nq,[])

compare(q,nq)
 
rmes = compare(f,nq)

%% 例8.7 利用无损预测和霍夫曼编码的混合IGS量化（8）
clc
clear
f = imread('..\Pictures\images_ch08\Fig0810(a)(Original).tif');
imshowMy(f)
title('原始图像')
imfinfoMy(f)

imshowMy(f, 8)
title('均匀量化为8级灰度图像')



q = quantize(f,3,'igs');
imshowMy(q,[])
imfinfoMy(q)
title('(改进的灰度级量化)IGS量化为8级灰度图像')

qs = double(q)/32;
imshowMy(qs,[])
imfinfoMy(qs)
title('IGS量化为8级灰度图像')

e = mat2lpc(qs);
c = mat2huff(e);
cr = imratio(f,c)
% ------------------
ne = huff2mat(c);
nqs = lpc2mat(ne);
% imshowMy(nqs)
nq = 32*nqs;
% imshowMy(nq,[])

compare(q,nq)
 
rmes = compare(f,nq)

%% 例8.7 利用无损预测和霍夫曼编码的混合IGS量化（32）
clc
clear
f = imread('..\Pictures\images_ch08\Fig0810(a)(Original).tif');
imshowMy(f)
title('原始图像')
imfinfoMy(f)

imshowMy(f, 32)
title('均匀量化为32级灰度图像')



q = quantize(f,5,'igs');
imshowMy(q,[])
imfinfoMy(q)
title('(改进的灰度级量化)IGS量化为32级灰度图像')

qs = double(q)/8;
imshowMy(qs,[])
imfinfoMy(qs)
title('IGS量化为32级灰度图像')

e = mat2lpc(qs);
c = mat2huff(e);
cr = imratio(f,c)
% ------------------
ne = huff2mat(c);
nqs = lpc2mat(ne);
% imshowMy(nqs)
nq = 8*nqs;
% imshowMy(nq,[])

compare(q,nq)
 
rmes = compare(f,nq)

%% 例8.8 im2jpeg JPEG 压缩 DCT 离散余弦
clc
clear

f = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy).tif');
imshowMy(f)

c1 = im2jpeg(f);
f1 = jpeg2im(c1);
imshowMy(f1)

imratio(f,c1)

compare(f,f1,3)

c4 = im2jpeg(f,4);
f4 = jpeg2im(c4);
imshowMy(f1)

imratio(f,c4)

compare(f,f4,3)

whos

%% 例8.8 im2jpeg JPEG 压缩 DCT 离散余弦(不显示图片)
clc
clear

f = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy).tif');

c1 = im2jpeg(f);
% f1 = jpeg2im(c1);
imratio(f,c1)

% compare(f,f1,3)

c4 = im2jpeg(f,4);
% f4 = jpeg2im(c4);
imratio(f,c4)

% compare(f,f4,3)

c8 = im2jpeg(f,8);
% f8 = jpeg2im(c8);
imratio(f,c8)

%% 例8.9
clc
clear
f = imread('..\Pictures\images_ch08\Fig0804(a)(Tracy).tif');
imshowMy(f)

c1 = im2jpeg2k(f,5,[8 8.5]);
f1 = jpeg2k2im(c1);
imshowMy(f1)
rms1 = compare(f,f1)
cr1 = imratio(f,c1)

c2 = im2jpeg2k(f,5,[8 7]);
f2 = jpeg2k2im(c2);
imshowMy(f2)
rms2 = compare(f,f2)
cr2 = imratio(f,c2)

c3 = im2jpeg2k(f,1,[1 1 1 1]);
f3 = jpeg2k2im(c3);
imshowMy(f3)
rms3 = compare(f,f3)
cr3 = imratio(f,c3)

whos

%%
clc
clear

      《六州歌头》
         -- 张孝祥 
  
长淮望断，关塞莽然平。 
征尘暗，霜风劲，悄边声。黯销凝。 
追想当年事，殆天数，非人力，洙泗上，弦歌地，亦膻腥。 
隔水毡乡，落日牛羊下，区脱纵横。 
看名王宵猎，骑火一川明。笳鼓悲鸣。遣人惊。 

念腰间箭、匣中剑，空埃蠹，竟何成。 
时易失，心徒壮，岁将零。渺神京。 
干羽方怀远，静烽燧，且休兵。 
冠盖使，纷驰鹜，若为情。 
闻道中原遗老，常南望，羽葆霓旌。 
使行人到此，忠愤气填膺，有泪如倾！ 






