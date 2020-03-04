highly_noise = imread('../../data/mri_image_noise_level_high.png');
low_noise = imread('../../data/mri_image_noise_level_low.png');
medium_noise = imread('../../data/mri_image_noise_level_medium.png');

noiseless = imread('../../data/mri_image_noiseless.png');
%imshow(m)
noiseless = double(noiseless);
noiseless = noiseless/255;

y = double(highly_noise);
y = y/255;
x_initial = y;

alpha = 0.24;
gamma = -1;

[final_xi_quad_1,obj_quad_high] = gradient_descent( x_initial , alpha , @quadratic,gamma);
[final_xi_quad_2,obj_quad_high2] = gradient_descent( x_initial , alpha*0.8 , @quadratic,gamma);
[final_xi_quad_3,obj_quad_high3] = gradient_descent( x_initial , alpha*1.2 , @quadratic,gamma);

RRMSE_quad_highnoise = norm(double(noiseless - final_xi_quad_1),'fro')/norm(double(noiseless),'fro');
RRMSE_quad_highnoise2 = norm(double(noiseless - final_xi_quad_2),'fro')/norm(double(noiseless),'fro');
RRMSE_quad_highnoise3 = norm(double(noiseless - final_xi_quad_3),'fro')/norm(double(noiseless),'fro');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');
fprintf('\n');

fprintf('Denoising High noise Using Quadratic Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('RRMSE(alpha) = %f\n', RRMSE_quad_highnoise);
fprintf('RRMSE(alpha*0.8) = %f\n' , RRMSE_quad_highnoise2);
fprintf('RRMSE(alpha*1.2) = %f\n' , RRMSE_quad_highnoise3);
fprintf('\n');


figure(1);
plot(obj_quad_high);
title('Quadratic Denoising in High Noise');
xlabel('No. of iterations');
ylabel('Cost Function')


alpha = 0.45;
gamma = 0.07;


[final_xi_huber_1,obj_huber_high] = gradient_descent( x_initial , alpha , @Huber,gamma);
[final_xi_huber_2,obj_huber_high2] = gradient_descent( x_initial , 0.8*alpha , @Huber,gamma);
[final_xi_huber_3,obj_huber_high3] = gradient_descent( x_initial , alpha , @Huber,0.8*gamma);
[final_xi_huber_4,obj_huber_high4] = gradient_descent( x_initial , 1.2*alpha , @Huber,gamma);
[final_xi_huber_5,obj_huber_high5] = gradient_descent( x_initial , alpha , @Huber,1.2*gamma);

RRMSE_huber_highnoise = norm(double(noiseless - final_xi_huber_1),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_highnoise2 = norm(double(noiseless - final_xi_huber_2),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_highnoise3 = norm(double(noiseless - final_xi_huber_3),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_highnoise4 = norm(double(noiseless - final_xi_huber_4),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_highnoise5 = norm(double(noiseless - final_xi_huber_5),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising High noise Using Huber Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', RRMSE_huber_highnoise);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , RRMSE_huber_highnoise2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , RRMSE_huber_highnoise4);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , RRMSE_huber_highnoise3);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , RRMSE_huber_highnoise5);

fprintf('\n');

figure(2);
plot(obj_huber_high);
title('Huber Denoising in high noise');
xlabel('No. of iterations');
ylabel('Cost Function')

alpha = 0.4;
gamma = 0.65;


[final_xi_disc_1,obj_disc_high] = gradient_descent( x_initial , alpha , @Disc,gamma);
[final_xi_disc_2,obj_disc_high2] = gradient_descent( x_initial , 0.8*alpha , @Disc,gamma);
[final_xi_disc_3,obj_disc_high3] = gradient_descent( x_initial , alpha , @Disc,0.8*gamma);
[final_xi_disc_4,obj_disc_high4] = gradient_descent( x_initial , 1.2*alpha , @Disc,gamma);
[final_xi_disc_5,obj_disc_high5] = gradient_descent( x_initial , alpha , @Disc,1.2*gamma);

RRMSE_disc_highnoise = norm(double(noiseless - final_xi_disc_1),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_highnoise2 = norm(double(noiseless - final_xi_disc_2),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_highnoise3 = norm(double(noiseless - final_xi_disc_3),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_highnoise4 = norm(double(noiseless - final_xi_disc_4),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_highnoise5 = norm(double(noiseless - final_xi_disc_5),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising High noise Using Disc Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', RRMSE_disc_highnoise);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , RRMSE_disc_highnoise2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , RRMSE_disc_highnoise4);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , RRMSE_disc_highnoise3);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , RRMSE_disc_highnoise5);

fprintf('\n');

figure(3);
plot(obj_disc_high);
title('LogPrior Denoising in high noise ');
xlabel('No. of iterations');
ylabel('Cost Function')


figure(4);

myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (noiseless)); 
colormap (myColorScale);
daspect ([1 1 1]);
axis tight;
colorbar
title('Noiseless Image');

figure(5);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (highly_noise)); 
colormap (myColorScale);
daspect ([1 1 1]);
axis tight;
colorbar
title('High Noise');



figure(6);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_quad_1));
colormap (myColorScale);
daspect ([1 1 1]);
axis tight;
colorbar
title('Quadratic Potential Denoised');

figure(7);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_huber_1)); 
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Huber Potential Denoised');

figure(8);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_disc_1)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Discontinuity Removal Denoised');


y = double(low_noise);
y = y/255;
x_initial = y;

alpha = 0.08;
gamma = -1;
[final_xi_quad_1,obj_quad_low] = gradient_descent( x_initial , alpha , @quadratic,gamma);
[final_xi_quad_2,obj_quad_low2] = gradient_descent( x_initial , alpha*0.8 , @quadratic,gamma);
[final_xi_quad_3,obj_quad_low3] = gradient_descent( x_initial , alpha*1.2 , @quadratic,gamma);

RRMSE_lownoise = norm(double(noiseless - final_xi_quad_1),'fro')/norm(double(noiseless),'fro');
RRMSE_quad_lownoise2 = norm(double(noiseless - final_xi_quad_2),'fro')/norm(double(noiseless),'fro');
RRMSE_quad_lownoise3 = norm(double(noiseless - final_xi_quad_3),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising Low noise Using Quadratic Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('RRMSE(alpha) = %f\n', RRMSE_lownoise);
fprintf('RRMSE(alpha*0.8) = %f\n' , RRMSE_quad_lownoise2);
fprintf('RRMSE(alpha*1.2) = %f\n' , RRMSE_quad_lownoise3);
fprintf('\n');

figure(9);
plot(obj_quad_low);
title('Quadratic Denoising in Low noise');
xlabel('No. of iterations');
ylabel('Cost Function')


alpha = 0.40;
gamma = 0.033;


[final_xi_huber_1,obj_huber_low] = gradient_descent( x_initial , alpha , @Huber,gamma);
[final_xi_huber_2,obj_huber_low2] = gradient_descent( x_initial , 0.8*alpha , @Huber,gamma);
[final_xi_huber_3,obj_huber_low3] = gradient_descent( x_initial , alpha , @Huber,0.8*gamma);
[final_xi_huber_4,obj_huber_low4] = gradient_descent( x_initial , 1.2*alpha , @Huber,gamma);
[final_xi_huber_5,obj_huber_low5] = gradient_descent( x_initial , alpha , @Huber,1.2*gamma);

RRMSE_huber_lownoise = norm(double(noiseless - final_xi_huber_1),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_lownoise2 = norm(double(noiseless - final_xi_huber_2),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_lownoise3 = norm(double(noiseless - final_xi_huber_3),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_lownoise4 = norm(double(noiseless - final_xi_huber_4),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_lownoise5 = norm(double(noiseless - final_xi_huber_5),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising Low noise Using Huber Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', RRMSE_huber_lownoise);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , RRMSE_huber_lownoise2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , RRMSE_huber_lownoise4);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , RRMSE_huber_lownoise3);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , RRMSE_huber_lownoise5);
fprintf('\n');

figure(10);
plot(obj_huber_low);
title('Huber Denoising in Low noise');
xlabel('No. of iterations');
ylabel('Cost Function')

alpha = 0.2;
gamma = 0.5;


[final_xi_disc_1,obj_disc_low] = gradient_descent( x_initial , alpha , @Disc,gamma);
[final_xi_disc_2,obj_disc_low2] = gradient_descent( x_initial , 0.8*alpha , @Disc,gamma);
[final_xi_disc_3,obj_disc_low3] = gradient_descent( x_initial , alpha , @Disc,0.8*gamma);
[final_xi_disc_4,obj_disc_low4] = gradient_descent( x_initial , 1.2*alpha , @Disc,gamma);
[final_xi_disc_5,obj_disc_low5] = gradient_descent( x_initial , alpha , @Disc,1.2*gamma);

RRMSE_disc_lownoise = norm(double(noiseless - final_xi_disc_1),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_lownoise2 = norm(double(noiseless - final_xi_disc_2),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_lownoise3 = norm(double(noiseless - final_xi_disc_3),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_lownoise4 = norm(double(noiseless - final_xi_disc_4),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_lownoise5 = norm(double(noiseless - final_xi_disc_5),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising Low noise Using Disc Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', RRMSE_disc_lownoise);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , RRMSE_disc_lownoise2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , RRMSE_disc_lownoise4);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , RRMSE_disc_lownoise3);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , RRMSE_disc_lownoise5);
fprintf('\n');

figure(11)
plot(obj_disc_low);
title('LogPrior Denoising in Low noise');
xlabel('No. of iterations');
ylabel('Cost Function')

figure(12);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (noiseless)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Noiseless');

figure(13);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (low_noise)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Low Noise');

figure(14);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_quad_1)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Quadratic Potential');

figure(15);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_huber_1)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Huber Potential');

figure(16);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_disc_1)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('LogPrior Potential');




y = double(medium_noise);
y = y/255;
x_initial = y;

alpha = 0.2;
gamma = -1;
[final_xi_quad_1,obj_quad_med] = gradient_descent( x_initial , alpha , @quadratic,gamma);
%optimal alpha for quadratic function was around 0.2. So showing RRMSE for
%0.24 and 0.16
[final_xi_quad_2,obj_quad_med2] = gradient_descent( x_initial , alpha*0.8 , @quadratic,gamma);
[final_xi_quad_3,obj_quad_med3] = gradient_descent( x_initial , alpha*1.2 , @quadratic,gamma);

RRMSE_quad_mednoise = norm(double(noiseless - final_xi_quad_1),'fro')/norm(double(noiseless),'fro');
RRMSE_quad_mednoise2 = norm(double(noiseless - final_xi_quad_2),'fro')/norm(double(noiseless),'fro');
RRMSE_quad_mednoise3 = norm(double(noiseless - final_xi_quad_3),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising Medium noise Using Quadratic Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('RRMSE(alpha) = %f\n', RRMSE_quad_mednoise);
fprintf('RRMSE(alpha*0.8) = %f\n' , RRMSE_quad_mednoise2);
fprintf('RRMSE(alpha*1.2) = %f\n' , RRMSE_quad_mednoise3);
fprintf('\n');

figure(17);
plot(obj_quad_med);
title('Quadratic Denoising using Medium Noise');
xlabel('No. of iterations');
ylabel('Cost Function');


alpha = 0.5;
gamma = 0.04;

[final_xi_huber_1,obj_huber_med] = gradient_descent( x_initial , alpha , @Huber,gamma);
[final_xi_huber_2,obj_huber_med2] = gradient_descent( x_initial , 0.8*alpha , @Huber,gamma);
[final_xi_huber_3,obj_huber_med3] = gradient_descent( x_initial , alpha , @Huber,0.8*gamma);
[final_xi_huber_4,obj_huber_med4] = gradient_descent( x_initial , 1.2*alpha , @Huber,gamma);
[final_xi_huber_5,obj_huber_med5] = gradient_descent( x_initial , alpha , @Huber,1.2*gamma);

RRMSE_huber_mednoise = norm(double(noiseless - final_xi_huber_1),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_mednoise2 = norm(double(noiseless - final_xi_huber_2),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_mednoise3 = norm(double(noiseless - final_xi_huber_3),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_mednoise4 = norm(double(noiseless - final_xi_huber_4),'fro')/norm(double(noiseless),'fro');
RRMSE_huber_mednoise5 = norm(double(noiseless - final_xi_huber_5),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising Medium noise Using Huber Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', RRMSE_huber_mednoise);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , RRMSE_huber_mednoise2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , RRMSE_huber_mednoise4);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , RRMSE_huber_mednoise3);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , RRMSE_huber_mednoise5);
fprintf('\n');

figure(18);
plot(obj_huber_med);
title('Huber Denoising using Medium Noise');
xlabel('No. of iterations');
ylabel('Cost Function');

alpha = 0.2;
gamma = 0.47;


[final_xi_disc_1,obj_disc_med] = gradient_descent( x_initial , alpha , @Disc,gamma);
[final_xi_disc_2,obj_disc_med2] = gradient_descent( x_initial , 0.8*alpha , @Disc,gamma);
[final_xi_disc_3,obj_disc_med3] = gradient_descent( x_initial , alpha , @Disc,0.8*gamma);
[final_xi_disc_4,obj_disc_med4] = gradient_descent( x_initial , 1.2*alpha , @Disc,gamma);
[final_xi_disc_5,obj_disc_med5] = gradient_descent( x_initial , alpha , @Disc,1.2*gamma);

RRMSE_disc_mednoise = norm(double(noiseless - final_xi_disc_1),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_mednoise2 = norm(double(noiseless - final_xi_disc_2),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_mednoise3 = norm(double(noiseless - final_xi_disc_3),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_mednoise4 = norm(double(noiseless - final_xi_disc_4),'fro')/norm(double(noiseless),'fro');
RRMSE_disc_mednoise5 = norm(double(noiseless - final_xi_disc_5),'fro')/norm(double(noiseless),'fro');

fprintf('Denoising Medium noise Using Disc Prior \n');
fprintf('Optimal Alpha = %f\n' , alpha);
fprintf('Optimal Gamma = %f\n' , gamma);
fprintf('RRMSE(alpha,gamma) = %f\n', RRMSE_disc_lownoise);
fprintf('RRMSE(alpha*0.8,gamma) = %f\n' , RRMSE_disc_lownoise2);
fprintf('RRMSE(alpha*1.2,gamma) = %f\n' , RRMSE_disc_lownoise4);
fprintf('RRMSE(alpha,gamma*0.8) = %f\n' , RRMSE_disc_lownoise3);
fprintf('RRMSE(alpha,gamma*1.2) = %f\n' , RRMSE_disc_lownoise5);
fprintf('\n');

figure(19);
plot(obj_disc_med);
title('LogPrior Denoising using Medium Noise');
xlabel('No. of iterations');
ylabel('Cost Function');


figure(20);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (noiseless)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Noiseless');


figure(21);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (medium_noise)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Medium Noise');

figure(22);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_quad_1)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Quadratic Potential');

figure(23);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_huber_1)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('Huber Potential');

figure(24);
myNumOfColors = 200;
myColorScale = [ [0:1/(myNumOfColors-1):1]' ,[0:1/(myNumOfColors-1):1]' , [0:1/(myNumOfColors-1):1]' ];
imagesc (single (final_xi_disc_1)); % phantom is a popular test image
colormap (myColorScale);
%colormap jet;
daspect ([1 1 1]);
axis tight;
colorbar
title('LogPrior Potential');


