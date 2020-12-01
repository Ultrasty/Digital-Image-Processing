%% 第二章 基本原理

%% imshow 解决动态范围较小问题
clc
clear

I = imread('..\Pictures\images_ch02\Fig0203(a)(chest-xray).tif');
figure,subplot(121),imshow(I),subplot(122),imhist(I) 
axis tight

% 动态范围较低
class(I)
min(I(:))  
max(I(:))

figure,imshow(I,[])



%% 保存图像
clc
clear

f = imread('..\Pictures\images_ch02\Fig0206(a)(rose-original).tif');
imshow(f)
print -f1 -dtiff -r300 hi_res_rose
% 保存到 G:\DIPUsingMATLAB\MFiles

%% im2uint8
clc
clear
f1 = [-0.5 0.5
     0.75 1.5]
g1 = im2uint8(f1)

f2 = uint8(f1*100)
g2 = im2uint8(f2)

f3 = f1*100
g3 = im2uint8(f3)

f4 = uint16(f1*50000)
g4 = im2uint8(f4)

%% im2bw 灰度图象变为二值图像
clc
clear

I = imread('liftingbody.png');
imshow(I)

BW = im2bw(I,0.46);

figure,imshow(BW)

%% imabsdiff 计算两幅图像间的绝对差
clc
clear

I = imread('cameraman.tif');
imshow(I,[])
J = uint8(filter2(fspecial('gaussian'), I));
imshow(J,[])
K = imabsdiff(I,J);
imshow(K,[]) % [] = scale data automatically

%% imcomplement 对图像求补
clc
clear

bw = imread('text.png');
bw2 = imcomplement(bw);
subplot(1,2,1),imshow(bw)
subplot(1,2,2),imshow(bw2)


%% im2double
clc
clear
f1 = [-0.5 0.5
     0.75 1.5]
g1 = im2double(f1)

f22 = uint8(f1)
f2 = uint8(f1*100)
g2 = im2double(f2)

f3 = f1*100
g3 = im2double(f3)

f4 = uint16(f1*50000)
g4 = im2double(f4)

%% 共轭转置、转置
clc
clear
f1 = [-0.5+3*i 0.5+2*i
     0.75 1.5]
f1'   % 共轭转置
f1.'  % 转置




%% 矩阵变成列向量、行向量（统统按列操作）
clc
clear
f1 = [-0.5+3*i 0.5+2*i
     0.75 1.5
     0.6 2+5*i]
f1(:)  % 矩阵变成列向量

f1(1:end) % 矩阵变成行向量

f1(1:2:end)

f1([1 4 5]) % 一个向量作为另一个向量的索引

%% linspace
clc
clear
x = linspace(0,1,5)

%% 一种极为有用的逻辑数组寻址!!!
clc
clear
A = [1 2 3
     4 5 6
     7 8 9]
I =[1 0 0
    0 0 1
    1 0 0];

D = logical(I)

A(D)
B = A
B(D) = 0;
B

%% magic是double类型
clc
clear

f1 = magic(5)
g1 = im2uint8(f1)

f2 = uint8(magic(5))
g2 = im2uint8(f2)




%% 
clc
clear

A = [1 2
     3 4 ]
B = A

B = 3
A

A = []
B



%%
clc
clear

A = [1 2
     3 4 ]
B = [8 9
    12 23]
A + B

A1 = uint8(A*100)
B1 = uint8(B*100)
A1 + B1

imadd(A1,B1)

%% max
clc
clear

A = [3 2
    0.6 0.5
     1 4 ]
[C,I] = max(A)
[C,I] = max(A,[],1)
[C,I] = max(A,[],2)



%%
clc
clear

eps

realmax

realmin

computer

version % MATLAB版本


error('你好，不要怕。')

A = [3 2
    0.6 0.5
     1 4 ]

%% twodsin 向量化算法优化时间
clc
clear
[rt,f,g] = twodsin(1,1/(4*pi),1/(4*pi),1024,1024);





%% meshgrid 极为重要的索引函数！！！
clc
clear

c = [1 2 3]
r = [1 2 3 4 5]
[C,R] = meshgrid(c,r)




%% input 交互式输入
clc
clear

reply = input('Do you want more? Y/N [Y]: ','s');
if isempty(reply)
    reply = 'Y';
end




%% input
clc
clear
reply = input('Enter your data: ','s')

class(reply)
size(reply)

n = str2num(reply)

class(n)
size(n)

%% strread
clc
clear

t = '12.6, x2y, z';
[a b c] = strread(t,'%f%q%q','delimiter',',')

%% str2num  请参见P43的经典例子

clc
clear

str2num(['1 2';'3 4'])



%% 结构
clc
clear

s.zxr = 5;
s.jie = 6;
s.child = '56h';

s
s.child
class(s.child)




%% Matlab中如何计算程序运行的时间？
clc
clear



        tic
          your_code;
        toc
        或者使用
        t=cputime; 
          your_operation; 
        cputime-t


%% tic toc

for n = 1:100
    A = rand(n,n);
    b = rand(n,1);
    tic
    x = A\b;
    t(n) = toc;
end
plot(t)


