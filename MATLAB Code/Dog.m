%% Import image and convert it into greyscale

img_dog = imread('Dog.jpeg');
img_dog = imresize(img_dog, 0.2);

img_3 = rgb2gray(img_dog);
img_3 = double(img_3);

[m, n] = size(img_3);

total = m*n; %count the total amount of pixels

brightness = img_3(:); %record the brightness level of each pixel from columns to columns


%% Parameter Setting

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
elapse_time = toc


%% Compute the degree matrix D

d = sum(W, 2);
d_invsqrt = d .^ (-1/2);

D = diag(d);
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


%% Plottting the scatter plot of the second smallest eigenvector

x = 1:1:total;
scatter(x, y_1, 5)
ax = gca;           % Get current axes handle
ax.FontSize = 20;
grid on;
yl = yline(optimal_split_pt,'--','Optimal Splitting Point','FontSize',23,'LineWidth',3);
title('Scatter Plot of the Second Smallest Eigenvector', 'FontSize', 30, 'FontWeight', 'bold');
xlabel('Position', 'FontSize', 30);
ylabel('Entry', 'FontSize', 30);


%% Eigenvector Density Clustering

band_width = 2.5e-4;  % Band width for scanning

min_y = min(y_1);
band_centers = (optimal_split_pt - band_width/2): -band_width : min_y


band_counts = zeros(length(band_centers), 1);

for i = 1:length(band_centers)
    lower = band_centers(i) - band_width/2;
    upper = band_centers(i) + band_width/2;
    band_counts(i) = sum(y_1 >= lower & y_1 < upper);
end

target_band_index = -1;
max_density = max(band_counts);

for j = 2:length(band_centers)-1
    if band_counts(j) > band_counts(j-1) && ...
       band_counts(j) > band_counts(j+1) && ...
       band_counts(j) > mean(band_counts(:, 1)) &&... % Identify the first band that is denser than its neighbors
       band_counts(j) > 0.7 * max_density % Also require it to be one of the densest bands globally
        target_band_index = j;
        break;
    end
end

new_split = band_centers(target_band_index) - band_width / 2


%% Plot the Result of Segmentation using 2 Splitting Points

maskA = reshape(y_1 > optimal_split_pt(1) | y_1 < new_split, m, n);  
maskB = reshape(y_1 <= optimal_split_pt(1) & y_1 >= new_split , m, n);

A = img_3 .* maskA;
B = img_3 .* maskB;

figure;
subplot(1, 3, 1);
imshow(uint8(img_3));
title('A', 'FontSize',25);

subplot(1, 3, 2);
imshow(uint8(A));
title('B', 'FontSize',25);

subplot(1, 3, 3);
imshow(uint8(B));
title('C', 'FontSize',25);


%% Plot the Result of Segmentation using Different Splitting Point Choices

maskA_1 = reshape(y_1 > 0 | y_1 < new_split, m, n);  
maskB_1 = reshape(y_1 <= 0 & y_1 >= new_split , m, n);

maskA_2 = reshape(y_1 > -2e-4 | y_1 < new_split, m, n);  
maskB_2 = reshape(y_1 <= -2e-4 & y_1 >= new_split , m, n);

maskA_3 = reshape(y_1 > optimal_split_pt(1) | y_1 < -4e-4, m, n);  
maskB_3 = reshape(y_1 <= optimal_split_pt(1) & y_1 >= -4e-4 , m, n);

maskA_4 = reshape(y_1 > optimal_split_pt(1) | y_1 < -3e-4, m, n);  
maskB_4 = reshape(y_1 <= optimal_split_pt(1) & y_1 >= -3e-4 , m, n);

B_1 = img_3 .* maskB_1;
B_2 = img_3 .* maskB_2;
B_3 = img_3 .* maskB_3;
B_4 = img_3 .* maskB_4;

figure;
subplot(2, 3, 1);
imshow(uint8(img_3));
title('A', 'FontSize',25);

subplot(2, 3, 2);
imshow(uint8(B));
title('B', 'FontSize',25);

subplot(2, 3, 3);
imshow(uint8(B_1));
title('C', 'FontSize',25);

subplot(2, 3, 4);
imshow(uint8(B_2));
title('D', 'FontSize',25);

subplot(2, 3, 5);
imshow(uint8(B_3));
title('E', 'FontSize',25);

subplot(2, 3, 6);
imshow(uint8(B_4));
title('F', 'FontSize',25);
