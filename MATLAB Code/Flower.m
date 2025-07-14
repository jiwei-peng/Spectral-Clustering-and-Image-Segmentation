%% Import image and convert it into greyscale

img_flower = imread('Flower.jpg');
img_flower = imresize(img_flower, 0.15);

img_2 = rgb2gray(img_flower);
img_2 = double(img_2);

[m, n] = size(img_2);

total = m*n; %count the total amount of pixels

brightness = img_2(:); %record the brightness level of each pixel from columns to columns


%% Parameter Setting 1

r = 11; 
sigma_I = 17.85;
sigma_X = 8; 


%% Constructing Adjancency Matrix for Parameter Setting 1

tic
W = sparse(total, total); %Create an empty adjacency matrix

for u = 1:total
    [u_i, u_j] = ind2sub([m, n], u);

    i_min = max(1, u_i - r);
    i_max = min(m, u_i + r);
    j_max = min(n, u_j + r);

    for v_i = i_min : i_max
        for v_j = u_j : j_max
            v = sub2ind([m, n], v_i, v_j);

            d_spatial = sqrt((u_i - v_i)^2 + (u_j - v_j)^2);
            
            if d_spatial < r && d_spatial > 0
                d_brightness = brightness(u) - brightness(v);
                w = exp(-(d_brightness^2)/sigma_I^2) * exp(-(d_spatial^2)/sigma_X^2);
                W(u, v) = w;
                W(v, u) = w;
            end
        end
    end
end

W; %Updated adjacency matrix
elapsed_time = toc


%% Compute the degree matrix D

d = sum(W, 2);
d_invsqrt = sum(W, 2) .^ (-1/2);

D = diag(sum(W, 2));
D_invsqrt = diag(d_invsqrt);

[V ~] = eigs(D_invsqrt * (D - W) * D_invsqrt, 2, "smallestabs");

y_1 = D .^ (1/2) \ V(:, 2);


%% Computing the optimal splitting point

split_pt = min(y_1): 1e-4 : max(y_1);
matrix_y = sparse(total, length(split_pt));
dsum = zeros(3, length(split_pt));
Ncut = zeros(length(split_pt), 1);

for u = split_pt
    idx = find(split_pt == u);
    for v = 1:total
        if y_1(v) > u
            matrix_y(v,idx) = 1;
            dsum(1, idx) = dsum(1, idx) + d(v);
        else
            matrix_y(v,idx) = -1;
            dsum(2, idx) = dsum(2, idx) + d(v);
        end
    end
end

dsum(3,:) = - dsum(1,:) ./ dsum(2,:);


for u = 1:length(split_pt)
    matrix_y(matrix_y(:, u) == -1, u) = dsum(3, u);
end


for i = 1:length(split_pt)
    Ncut(i) = (matrix_y(:, i)' * (D - W) * matrix_y(:, i))/ ...
        (matrix_y(:, i)' * D * matrix_y(:, i));
end

optimal_split_pt = split_pt(find(Ncut == min(Ncut)))


%%  Store Splitted Regions through Binary Representations

maskA1 = reshape(y_1 > optimal_split_pt(1), m, n);  
maskB1 = reshape(y_1 <= optimal_split_pt(1), m, n);  


%% Parameter Setting 2

r = 8; 
sigma_I = 13; 
sigma_X = 7; 


%%  Constructing Adjancency Matrix for Parameter Setting 2

tic
W = sparse(total, total); %Create an empty adjacency matrix

for u = 1:total
    [u_i, u_j] = ind2sub([m, n], u);

    i_min = max(1, u_i - r);
    i_max = min(m, u_i + r);
    j_max = min(n, u_j + r);

    for v_i = i_min : i_max
        for v_j = u_j : j_max
            v = sub2ind([m, n], v_i, v_j);

            d_spatial = sqrt((u_i - v_i)^2 + (u_j - v_j)^2);
            
            if d_spatial < r && d_spatial > 0
                d_brightness = brightness(u) - brightness(v);
                w = exp(-(d_brightness^2)/sigma_I^2) * exp(-(d_spatial^2)/sigma_X^2);
                W(u, v) = w;
                W(v, u) = w;
            end
        end
    end
end

W; %Updated adjacency matrix
elapsed_time = toc


%% Compute the degree matrix D

d = sum(W, 2);
d_invsqrt = sum(W, 2) .^ (-1/2);

D = diag(sum(W, 2));
D_invsqrt = diag(d_invsqrt);

[V ~] = eigs(D_invsqrt * (D - W) * D_invsqrt, 3, "smallestabs");

new_y_1 = D .^ (1/2) \ V(:, 2);


%% Computing the optimal splitting point

split_pt = min(new_y_1): 1e-4 : max(new_y_1);
matrix_y = sparse(total, length(split_pt));
dsum = zeros(3, size(matrix_y, 2));
Ncut = zeros(length(split_pt), 1);

for u = split_pt
    idx = find(split_pt == u);
    for v = 1:total
        if new_y_1(v) > u
            matrix_y(v,idx) = 1;
            dsum(1, idx) = dsum(1, idx) + d(v);
        else
            matrix_y(v,idx) = -1;
            dsum(2, idx) = dsum(2, idx) + d(v);
        end
    end
end

dsum(3,:) = -dsum(1,:) ./ dsum(2,:);


for u = 1:length(split_pt)
    matrix_y(matrix_y(:, u) == -1, u) = dsum(3, u);
end


for i = 1:length(split_pt)
    Ncut(i) = (matrix_y(:, i)' * (D - W) * matrix_y(:, i))/ ...
        (matrix_y(:, i)' * D * matrix_y(:, i));
end

new_optimal_split_pt = split_pt(find(Ncut == min(Ncut)))


%% Plot the Result of Segmentation with 2 Sets of Parameters

maskA2 = reshape(new_y_1 > new_optimal_split_pt(1), m, n);
maskB2 = reshape(new_y_1 <= new_optimal_split_pt(1), m, n);

A1 = img_2 .* maskA1;
B1 = img_2 .* maskB1;
A2 = img_2 .* maskA2;
B2 = img_2 .* maskB2;

figure;
subplot(2, 3, 1);
imshow(uint8(img_2));
title('A','FontSize', 25);

subplot(2, 3, 2);
imshow(uint8(A1));
title('B','FontSize', 25);

subplot(2, 3, 3);
imshow(uint8(B1));
title('C','FontSize', 25);

subplot(2, 3, 5);
imshow(uint8(A2));
title('D','FontSize', 25);

subplot(2, 3, 6);
imshow(uint8(B2));
title('E','FontSize', 25);


%% Thresholding

y_1_sigma = ones(size(y_1));
y_1_sigma (y_1 <= optimal_split_pt(1)) = -1;

new_y_1_sigma = ones(size(new_y_1));
new_y_1_sigma (new_y_1 <= new_optimal_split_pt(1)) = -1;

norm_diff = (norm(0.5 * (new_y_1_sigma - y_1_sigma))^2) / (m*n)

