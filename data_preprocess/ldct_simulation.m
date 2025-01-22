close all; 
clear all; 
clc;

run('./irt/setup.m')

%% system setup
detect_number = 672;
angle = 1152;
pixel_size = 512;
sg = sino_geom( ...
    'fan','nb', detect_number, 'na',angle, ...
    'ds', 1.85, 'dsd', 1361.2, 'dod',615.18 , ...
    'source_offset',0.0,'orbit',360, 'down', 1);
ig = image_geom('nx',pixel_size, 'ny', pixel_size,'fov',350,'offset_x',0,'down', 1);
G = Gtomo2_dscmex(sg, ig);

%% scan-parameter setup
u_water = 0.0192;
I0 = 1.5e5;
Dose = [1/2 1/4 1/10 1/20];
[~, m] = size(Dose);

%% patient_id
patient_id = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506];

[~,t] = size(patient_id);
for i = 1:t
    id = patient_id(i);
    disp(id)
    % MAT file is obtained by converting from original DICOM file.
    ref_name = strcat('./AAPM_LDCT_mat/L',num2str(id,'%03d'),'_full_1mm_CT.mat');
    load(ref_name);
    [h,w,l] = size(Img_CT);
    disp(l)

    img = zeros(1, 1, pixel_size, pixel_size);
    sino = zeros(1, 1, detect_number, angle);

    num = 0;
    for j = 1:l
        num = num+1;
        xtrue = double(Img_CT(:,:,j));
        xtrue = ((xtrue-1024)/1000)*u_water+u_water;
        hsino = G * xtrue;
        for k = 1:m
            dose = Dose(k);
            sig = 10; % standard variance of electronic noise, a characteristic of CT scanner
            
            % perform low-dose simulation on the projection data
            kappa = 1;
            ri = 1.0;
            yb = dose * I0 .* exp(-hsino/kappa) + ri; % exponential transform to incident flux
            
            yi = poisson(yb) + sqrt(sig)*randn(size(yb));

            li_hat = -log((yi-ri)./(I0*dose));
            li_hat(yi-ri <= 0) = 0;
            lsino = li_hat * kappa;
            sino(k,num,:,:) = lsino;
            
            % reconstruct image
            tmp = fbp2(sg, ig);
            fbp = fbp2(lsino, tmp);
            img(k,num,:,:) = fbp;
        end
    end
    
    save_sino = strcat('./gen_data/mayo_2016_sim_mat/L',num2str(id,'%03d'),'_sino.mat');
    save_img = strcat('./gen_data/mayo_2016_sim_mat/L',num2str(id,'%03d'),'_img.mat');
    
    save(save_sino, 'sino','-v7.3');
    save(save_img, 'img','-v7.3');
end