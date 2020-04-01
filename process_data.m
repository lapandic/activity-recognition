%% Data is imported from text files

data_walking = importdata('data/walking_gyro-accel-gps20200328T172332.txt','\t');
acc_walking = data_walking.data(data_walking.textdata(:,2) == "ACC",:);

data_running = importdata('data/running_gyro-accel-gps20200328T173450.txt','\t');
acc_running = data_running.data(data_running.textdata(:,2) == "ACC",:);

data_standing = importdata('data/standing_gyro-accel-gps20200328T172058.txt','\t');
acc_standing = data_standing.data(data_standing.textdata(:,2) == "ACC",:);

% cut off the tail
%acc_walking = acc_walking(1:7200,:);
%acc_running = acc_running(1:6300,:);
%acc_standing = acc_standing(1:8100,:);

freq = 100;
T = 1/freq;

[N_walking,d] = size(acc_walking);
t_walking = [0:T:(N_walking-1)*T]';

[N_running,d] = size(acc_running);
t_running = [0:T:(N_running-1)*T]';

[N_standing,d] = size(acc_standing);
t_standing = [0:T:(N_standing-1)*T]';

%% training-test data split
window_length = 300;
N = [N_standing, N_walking, N_running];
num_training_samples = N*0.8;
num_windows = floor(N./window_length);
training_windows = floor(num_training_samples./window_length);
test_windows = num_windows-training_windows;

%training data
Training_standing = zeros(training_windows(1),window_length,3);
Training_walking = zeros(training_windows(2),window_length,3);
Training_running = zeros(training_windows(3),window_length,3);

Training_standing_t = zeros(training_windows(1),window_length,1);
Training_walking_t = zeros(training_windows(2),window_length,1);
Training_running_t = zeros(training_windows(3),window_length,1);

for i=1:training_windows(1)
	Training_standing(i,:,:) = acc_standing((i-1)*300+1:i*300,:);
    Training_standing_t(i,:,:) = t_standing((i-1)*300+1:i*300,:);
end


for i=1:training_windows(2)
	Training_walking(i,:,:) = acc_walking((i-1)*300+1:i*300,:);
	Training_walking_t(i,:,:) = t_walking((i-1)*300+1:i*300,:);
end

for i=1:training_windows(3)
	Training_running(i,:,:) = acc_running((i-1)*300+1:i*300,:);
	Training_running_t(i,:,:) = t_running((i-1)*300+1:i*300,:);
end

% test data
Test_standing = zeros(test_windows(1),window_length,3);
Test_walking = zeros(test_windows(2),window_length,3);
Test_running = zeros(test_windows(3),window_length,3);

Test_standing_t = zeros(test_windows(1),window_length,1);
Test_walking_t = zeros(test_windows(2),window_length,1);
Test_running_t = zeros(test_windows(3),window_length,1);

for i=training_windows(1)+1:num_windows(1)
	Test_standing(i-training_windows(1),:,:) = acc_standing((i-1)*300+1:i*300,:);
	Test_standing_t(i-training_windows(1),:,:) = t_standing((i-1)*300+1:i*300,:);    
end

for i=training_windows(2)+1:num_windows(2)
	Test_walking(i-training_windows(2),:,:) = acc_walking((i-1)*300+1:i*300,:);
	Test_walking_t(i-training_windows(2),:,:) = t_walking((i-1)*300+1:i*300,:);
end

for i=training_windows(3)+1:num_windows(3)
	Test_running(i-training_windows(3),:,:) = acc_running((i-1)*300+1:i*300,:);
    Test_running_t(i-training_windows(3),:,:) = t_running((i-1)*300+1:i*300,:);
end

Test_data = [Test_standing;Test_walking;Test_running];
Test_data_t = [Test_standing_t; Test_walking_t; Test_running_t];

%% plot to analyze
plot_data = 0;
if plot_data == 1
    figure(1)
    plot(t_walking,acc_walking)
    legend('x','y','z');
    grid on;
    title('walking');

    figure(2)
    plot(t_standing,acc_standing)
    legend('x','y','z');
    grid on;
    title('standing');

    figure(3)
    plot(t_running,acc_running)
    legend('x','y','z');
    grid on;
    title('running');
end
%% create knn model
dim_of_vector = 21;
idx_swr = [ones(training_windows(1),1);2*ones(training_windows(2),1);3*ones(training_windows(3),1)];
feature_vectors = zeros(length(idx_swr),dim_of_vector);
for i=1:training_windows(1)
    feature_vectors(i,:) = get_feature_vector(squeeze(Training_standing(i,:,:)));
end

for i=1:training_windows(2)
    feature_vectors(training_windows(1)+i,:) = get_feature_vector(squeeze(Training_walking(i,:,:)));
end

for i=1:training_windows(3)
    feature_vectors(training_windows(1)+training_windows(2)+i,:) = get_feature_vector(squeeze(Training_running(i,:,:)));
end


%% calculate knn
size_td = size(Test_data);
test_data_classes = [ones(test_windows(1),1);2*ones(test_windows(2),1);3*ones(test_windows(3),1)];
for i=1:size_td(1)
    % perform knn search in the space of feature_vectors 
    ids = knnsearch(feature_vectors,get_feature_vector(squeeze(Test_data(i,:,:))),'K',5);

    % match indices of the obtained feature vectors to classes
    votes = idx_swr(ids);
    % calculate votes
    votes_per_class = [sum(votes(:) == 1),sum(votes(:) == 2),sum(votes(:) == 3)];
    % get the class with max number of votes
    [maxs,class] = max(votes_per_class);
    
    disp("-----------");
    fprintf("Test sample: %d\n",i);
    if class == 1
        pred_result = "Standing";
    elseif class == 2
        pred_result = "Walking";
    elseif class == 3
        pred_result = "Running";
    end
    
    expected_class = test_data_classes(i);
    if expected_class == 1
        exp_result = "Standing";
    elseif expected_class == 2
        exp_result = "Walking";
    elseif expected_class == 3
        exp_result = "Running";
    end
    
    disp(["Predicted result: " + pred_result]);
    disp(["Expected result: " + exp_result]);
    
    if plot_data
        figure(3+i)
        data_t = Test_data_t(i, :, :)';
        test_data = reshape(Test_data(i, :, :), size(Test_data, 2), size(Test_data, 3));
        plot(data_t, test_data)
        legend('x','y','z');
        grid on;
        title("Predicted: " + pred_result + ", Expected: " + exp_result);
    end
end
    

%% function

function [vec] = get_feature_vector(signal)
    vec = zeros(1,21);
    mag = sum(signal.^2,2);
   
    vec(1) = std(mag);
    vec(2) = mean(mag);
    vec(3) = min(mag);
    vec(4) = max(mag);
    vec(5) = prctile(mag,10);
    vec(6) = prctile(mag,25);
    vec(7) = prctile(mag,50);
    vec(8) = prctile(mag,75);
    vec(9) = prctile(mag,90);
    vec(10) = sum(mag(mag > prctile(mag,5)));
    vec(11) = sum(mag(mag > prctile(mag,10)));
    vec(12) = sum(mag(mag > prctile(mag,25)));
    vec(13) = sum(mag(mag > prctile(mag,75)));
    vec(14) = sum(mag(mag > prctile(mag,90)));
    vec(15) = sum(mag(mag > prctile(mag,95)));
    vec(16) = sum(mag(mag > prctile(mag,5)).^2);
    vec(17) = sum(mag(mag > prctile(mag,10)).^2);
    vec(18) = sum(mag(mag > prctile(mag,25)).^2);
    vec(19) = sum(mag(mag > prctile(mag,75)).^2);
    vec(20) = sum(mag(mag > prctile(mag,90)).^2);
    vec(21) = sum(mag(mag > prctile(mag,95)).^2);
end




