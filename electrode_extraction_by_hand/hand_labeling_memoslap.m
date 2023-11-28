%%% process and hand label UTE images.
%%% requires a head mask (ute_mask.nii.gz) and a 1mm isotropic UTE image
%%% requires the image processing toolbox (imdilate, imerode, imfilter, as
%%% well as some other small functions which should be included in the
%%% package, also requires the matlab nifti package by Jimmy Shen
%%% outputs a bunch of stuff, but the original hand-labeled coordinates are
%%% saved in mricoords_1, and are in "voxel space" of the UTE which was
%%% input to the program, ie each coordinate is a voxel offset along that
%%% axis (x,y or z).

%% author Russell Butler russell.buttler@usherbrooke.ca

clear all; close all;

sub_to_analyse = 'sub-010'; %insert here
session =2; % and here
run = 1; % and here

if ~ismember(session, 1:4) 
    error("`session` must be an integer between 1 and 4")
end

electrode_dir = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction/';

sub_dir = strcat(electrode_dir, sub_to_analyse, '/');

% paths to relevant data and output
path_ute = strcat(sub_dir, 'unzipped/', 'r', sub_to_analyse, ...
    '_ses-', num2str(session) , '_acq-petra_run-0', ...
    num2str(run),'_PDw.nii');
path_mask = strcat(sub_dir ,'/mask/test_mask_smooth_fwhm_4.nii.gz');

if ~isfile(path_ute)
    error(strcat("Petra not found. Check if the path is correct.\n", ...
        " Path: ", path_ute))
end

[filepath, name, ext] = fileparts(path_ute);

if ~startsWith(name, 'r')
    error('Petra file name must start with "r". Check if you choose a coregistered file.')
end

path_output = strcat(sub_dir, 'electrode_extraction/', 'ses-', ...
    num2str(session), '/', 'run-0', num2str(run), '/');

% the order in which to label the electrodes. after each mouse click, the
% program will move to the next electrode, ie, each mouse click labels an
% electrode in the sequence "elecorder" (wait for the crosshair)
% it is recommended that you put a gel capsule on the cap, before the
% experiment so you can be 100% sure which side is left and which is right,
% as this might get swapped depending on the converter you use
%elecorder = {'FP1','FPZ','FP2','AF8','AF4','GND','AF3','AF7','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT10','FT8','FC6','FC4','FC2','REF','FC1','FC3','FC5','FT7','FT9',...
%    'T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP10','TP8','CP6','CP4','CP2','CPZ','CP1','CP3','CP5','TP7','TP9','P7','P5','P3','P1','PZ','P2','P4','P6','P8',...
%    'PO8','PO4','POZ','PO3','PO7','O1','OZ','O2'} ;

elecorder = {'anode','a_r1','a_r2','a_r3', 'cathode_1','c1_r1','c1_r2','c1_r3', 'cathode_2','c2_r1','c2_r2','c2_r3', 'cathode_3','c3_r1','c3_r2','c3_r3'};

% get the raw UTE, and the mask
disp('loading raw data...');
rute = load_untouch_nii(path_ute); ruteorig = double(rute.img); % raw ute
mask = load_untouch_nii(path_mask); % mask

%if isfolder(path_output) 
%    error("Output directory already exists. Check if the electrodes have already been extracted.")
%end

mkdir(path_output)
% save UTE and mask in the output directory, for segmentation script
save_untouch_nii(rute, strcat(path_output, 'petra_.nii.gz'));
% BUG: somehow the finalmask only gets saved, when you save it twice
save_untouch_nii(mask, strcat(path_output, 'finalmask.nii.gz'));
save_untouch_nii(mask, strcat(path_output, 'finalmask.nii.gz'));
%% create the layers surrounding the head mask (12 layers)
mask.img = ceil(mask.img);
layer2 = imdilate(mask.img == 2, strel(ones(5, 5, 5)));
prevdil = double(mask.img == 1);
outim = double(mask.img == 0);
disp('performing iterative dilation...');
intensitylayers = zeros(size(prevdil));

n_layers = 12;
kernel_i = 3;
kernel = ones(kernel_i, kernel_i, kernel_i);

for i = 1:n_layers
    dilmaski = imdilate(prevdil, strel(kernel)) .* outim;
    clayer = (dilmaski - prevdil) > 0;
    intensitylayers(clayer == 1) = i;
    
    if i == 1
        prevdil = dilmaski;
    else
        prevdil = (prevdil + dilmaski) > 0;
    end
    
    intensitylayers = intensitylayers .* (~layer2);
end

disp('computing final layers...');
layers = intensitylayers;
dilbottom = imdilate(mask.img == 2, strel(ones(n_layers, n_layers, n_layers)));
layers(dilbottom == 1) = 0;
inds = find(layers > 0);

% save the layers
rute.img = single(layers);
save_untouch_nii(rute, strcat(path_output, 'layers.nii.gz'));

layers_binarized = rute;
layers_binarized.img(layers_binarized.img > 0) = 1;

save_untouch_nii(layers_binarized, strcat(path_output, ...
    'layers_binarized.nii.gz'));

rute_masked = layers_binarized;
rute_masked.img = rute_masked.img .* ruteorig;

save_untouch_nii(rute_masked, strcat(path_output,...
    'petra_masked.nii.gz'));


% do the pancake projection
disp('performing pancake projection...');
gsimg = double(ruteorig - imfilter(rute.img, ...
    fspecial('gaussian', 61, 61))); % first, take the gradient of the raw UTE (so the electrodes show up more clearly)
max_xp = 2.5; min_xp = -2.5;
max_yp = 2.5; min_yp = -2.5; % hard-coded grid variables
nsteps = 550;

max_layer_idx = 6;

for layer_index = 1:max_layer_idx
    layerinds = find(layers == layer_index);
    [lx, ly, lz] = ind2sub(size(layers), layerinds);
    [cx, cy, cz] = centmass3(layers);
    xlayer_diffs = lx - cx; ylayer_diffs = ly - cy; zlayer_diffs = lz - cz;
    [theta, phi, rho] = cart2sph(xlayer_diffs, ylayer_diffs, zlayer_diffs);
    if layer_index == 1; ftheta = theta; fphi = phi; frho = rho; end
    theta_brain = zeros(size(layers)); phi_brain = zeros(size(layers));
    rho_brain = zeros(size(layers));
    theta_brain(layerinds) = theta; phi_brain(layerinds) = phi;
    rho_brain(layerinds) = rho;
    [pancake_x, pancake_y] = pol2cart(theta, max(phi) - phi); % theta and phi are the rotation point and height point (z) of the head
    xsteps = min_xp:(max_xp - min_xp) / nsteps:(max_xp);
    ysteps = min_yp:(max_yp - min_yp) / nsteps:(max_yp);
    [xg, yg] = meshgrid(xsteps, ysteps);
    vq = griddata(double(pancake_x), double(pancake_y), ...
        gsimg(layerinds), double(xg), double(yg));
    vq(isnan(vq)) = 0;
    vqs(layer_index, :, :) = vq;
end

for i = 1:size(vqs, 1)
    dqs(i, :, :) = mat2gray((squeeze(vqs(i, :, :))) - imfilter((squeeze(vqs(i, :, :))), fspecial('gaussian', 60, 30)));
end

% interpolate super high intensity pixels (z > 5), for improved image contrast
% first find the bad pixels
th = 5;
n_steps_plus_one = nsteps + 1;

for i = 1:max_layer_idx
    dqi = squeeze(dqs(i, :, :));
    za = (reshape(dqi, [1, n_steps_plus_one * n_steps_plus_one]));
    zainds = find((za ~= 0)); zvals = zscore(za(zainds));
    bads = find(zvals > th | zvals <- th);
    za(zainds(bads)) = 0;
    dqs(i, :, :) = reshape(za, [n_steps_plus_one, n_steps_plus_one]);
end

% then do the interpolation
for i = 1:max_layer_idx
    [xind, yind] = ind2sub(size(squeeze(dqs(i, :, :))), ...
        find(squeeze(dqs(i, :, :)) ~= 0));
    dqsi = squeeze(dqs(i, :, :));
    vals = dqsi((squeeze(dqs(i, :, :)) ~= 0));
    [xg1, yg1] = meshgrid(1:n_steps_plus_one, 1:n_steps_plus_one);
    itp = griddata(xind, yind, vals, xg1, yg1);
    idqs(i, :, :) = (itp)';
end

% create the RGB pancake
im1 = (uint8(mat2gray(squeeze(mean(idqs(5:6, :, :), 1))) * 255));
im2 = (uint8(mat2gray(squeeze(mean(idqs(3:4, :, :), 1))) * 255));
im3 = (uint8(mat2gray(squeeze(mean(idqs(1:2, :, :), 1))) * 255));
expon = 1.5; % to accentuate higher value pixels
rgbs(:, :, 3) = uint8(mat2gray(im1) .^ expon * 255);
rgbs(:, :, 2) = uint8(mat2gray(im2) .^ expon * 255);
rgbs(:, :, 1) = uint8(mat2gray(im3) .^ expon * 255);
imwrite(rgbs, strcat(path_output, 'rgbs.png'));
save_nii(make_nii(idqs(1:6, :, :)), strcat(path_output, 'idqs.nii.gz'));

maskvq = zeros(size(vqs));
maskvq(vqs ~= 0) = 1;

save_nii(make_nii(maskvq(1:5, :, :)), strcat(path_output, 'maskvq.nii.gz'));

fhandle = figure('Position', [10, -10, 1000, 1000]);
imagesc(rgbs);
n = length(elecorder); coordinates = zeros(n, 2); hold on;

for i = 1:n
    title(elecorder{i});
    [x, y] = ginput(1);
    coordinates(i, :) = [x, y];
    plot(coordinates(:, 1), coordinates(:, 2), '.', 'Color', [1, 1, 0], 'LineWidth', 2);
    text(x, y, elecorder{i}, 'color', 'w', 'Fontsize', 12);
end

hold off

% randomize the coordinates, and invert them to 3d MRI space (the original
% hand-labeled coordinates are in mricoords_1)
for mricoordn = 1:30
    
    if mricoordn == 1 % save the original coordinates
        randcoords(:, 1) = coordinates(:, 1);
        randcoords(:, 2) = coordinates(:, 2);
    else
        randcoords(:, 1) = coordinates(:, 1) + (rand(1, n)' - .5) * 10; % random offset (produces ~1.5mm standard deviation in 3d MRI space)
        randcoords(:, 2) = coordinates(:, 2) + (rand(1, n)' - .5) * 10;
    end
    
    roundcoords = round(randcoords);
    for i = 1:size(roundcoords, 1)
        xgvals(i) = xg(roundcoords(i, 2), roundcoords(i, 1));
        ygvals(i) = yg(roundcoords(i, 2), roundcoords(i, 1));
    end
    [inv_theta, inv_phi] = cart2pol(xgvals, ygvals);
    inv_phi = 1.5708 - inv_phi;
    
    for elec = 1:size(roundcoords, 1)
        oz_theta = inv_theta(elec); oz_phi = inv_phi(elec);
        sumdiffs = sqrt((ftheta - oz_theta) .^ 2 + (fphi - oz_phi) .^ 2);
        minsumdiff_index = find(sumdiffs == min(sumdiffs), 1); % index of the closest theta,phi point
        min_rho = frho(minsumdiff_index);
        min_phi = fphi(minsumdiff_index);
        min_theta = ftheta(minsumdiff_index);
        [x, y, z] = sph2cart(min_theta, min_phi, min_rho);
        ex(elec) = x + cx; ey(elec) = y + cy; ez(elec) = z + cz;
    end
    
    elecimg = zeros(size(gsimg));
    colorelecs = zeros(size(gsimg));
    ex = round(ex); ey = round(ey); ez = round(ez);
    
    for coord = 1:size(coordinates, 1)
        elecimg(ex(coord), ey(coord), ez(coord)) = 1000;
        colorelecs(ex(coord), ey(coord), ez(coord)) = coord;
    end
    
    mricoords = [ex; ey; ez]; disp(mricoords);
    save([path_output, '/', 'mricoords_', num2str(mricoordn)], 'mricoords');
    
    if mricoordn == 1
        writematrix(mricoords, strcat(path_output, "handextracted_electrode_pos.csv"))
    end
end
