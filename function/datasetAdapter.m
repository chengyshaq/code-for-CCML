clc;
addpath(genpath('./'));
directory = dir('data2/*.mat');
items = {directory.name};

count = size(items, 2);
for i = 1 : 2 : count
    j = i + 1;
    test = load(items{i});
    train = load(items{j});
    
    test_data = test.test_data;
    test_target = test.test_target;
    train_data = train.train_data;
    train_target = train.train_target;
    
    fileName = items(i);
    charArray = fileName{1, 1};
    strs = strsplit(charArray, "_");
    str = "bindData\\" + strs(1) + ".mat";
    save(str, "test_data", "test_target", "train_data", "train_target");
end

clc;
load("corel5k");

    test_data = test.test_data;
    test_target = test.test_target;
    train_data = train.train_data;
    train_target = train.train_target;

count = size(items, 2);
for i = 1 : 2 : count
    j = i + 1;
    test = load(items{i});
    train = load(items{j});
    
    test_data = test.test_data;
    test_target = test.test_target;
    train_data = train.train_data;
    train_target = train.train_target;
    
    fileName = items(i);
    charArray = fileName{1, 1};
    strs = strsplit(charArray, "_");
    str = "bindData\\" + strs(1) + ".mat";
    save(str, "test_data", "test_target", "train_data", "train_target");
end