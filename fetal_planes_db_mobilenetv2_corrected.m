%% FedXAI-BrainNet - Corrected real-data experiment on FETAL_PLANES_DB (Section 6.7)
% -------------------------------------------------------------------------
% Fixes the labelling defect of the earlier pilot, in which every image was
% given the single folder-derived label 'Images'. Here class labels are read
% from the dataset's metadata file, not from folder names.
%
%   Dataset : Burgos-Artizzu et al., Sci. Rep. 2020 [9]
%             https://doi.org/10.5281/zenodo.3904280
%             Unzip so that DATA_ROOT contains FETAL_PLANES_DB_data.csv
%             (or .xlsx) and the folder Images/ with the .png files.
%
%   What it does
%     1. Keeps only fetal-brain images whose brain-plane label is
%        Trans-thalamic, Trans-cerebellum or Trans-ventricular (3 classes).
%     2. Uses the training/test split distributed with the dataset
%        (patient-disjoint); a patient-grouped 10% of training is held out
%        for validation.
%     3. Fine-tunes ImageNet MobileNetV2 (transfer learning).
%     4. Reports accuracy, macro precision/recall/F1, confusion matrix.
%     5. Reports test accuracy per ultrasound machine (device robustness).
%     6. Optional: leave-one-machine-out training (cross-device
%        generalization axis of Table 3). Set RUN_LOMO = true.
%
%   Requires: Deep Learning Toolbox, Image Processing Toolbox, and the
%   "Deep Learning Toolbox Model for MobileNet-v2 Network" support package.
% -------------------------------------------------------------------------
clear; clc; rng(42);

%% ---------------- User settings ----------------
DATA_ROOT  = 'D:\Datasets\FETAL_PLANES_ZENODO';   % <-- change this
IMG_DIR    = fullfile(DATA_ROOT, 'Images');
OUT_DIR    = fullfile(pwd, 'results_fetal_planes_db');
EPOCHS     = 15;
BATCH      = 32;
LR         = 1e-4;
RUN_LOMO   = false;   % leave-one-machine-out experiment (slow: one training per machine)

% Column names in the metadata file. If readtable reports different names,
% edit these four lines (the script prints the names it finds).
COL_IMG     = 'Image_name';
COL_PATIENT = 'Patient_num';
COL_PLANE   = 'Plane';
COL_BRAIN   = 'Brain_plane';
COL_MACHINE = 'US_Machine';
COL_TRAIN   = 'Train';
KEEP_CLASSES = {'Trans-thalamic','Trans-cerebellum','Trans-ventricular'};

if ~exist(OUT_DIR,'dir'); mkdir(OUT_DIR); end

%% ---------------- 1. Read metadata ----------------
metaFile = fullfile(DATA_ROOT, 'FETAL_PLANES_DB_data.csv');
if ~isfile(metaFile); metaFile = fullfile(DATA_ROOT, 'FETAL_PLANES_DB_data.xlsx'); end
if endsWith(metaFile, '.csv')
    opts = detectImportOptions(metaFile, 'Delimiter', ';');
    opts = setvartype(opts, 'char');
    T = readtable(metaFile, opts);
else
    T = readtable(metaFile, 'TextType', 'char');
end
fprintf('Metadata columns found: %s\n', strjoin(T.Properties.VariableNames, ', '));

plane   = strtrim(string(T.(COL_PLANE)));
brain   = strtrim(string(T.(COL_BRAIN)));
isBrain = contains(lower(plane), 'brain') & ismember(brain, string(KEEP_CLASSES));
T = T(isBrain, :);
brain = brain(isBrain);

files   = fullfile(IMG_DIR, strcat(string(T.(COL_IMG)), '.png'));
labels  = categorical(brain, string(KEEP_CLASSES));
patient = string(T.(COL_PATIENT));
machine = categorical(strtrim(string(T.(COL_MACHINE))));
isTrain = str2double(string(T.(COL_TRAIN))) == 1;

missing = ~isfile(files);
if any(missing)
    warning('%d image files not found and will be skipped.', nnz(missing));
    files = files(~missing); labels = labels(~missing); patient = patient(~missing);
    machine = machine(~missing); isTrain = isTrain(~missing);
end

% Guard against the earlier defect: there must be exactly 3 classes.
assert(numel(categories(removecats(labels))) == 3, ...
    'Expected 3 brain-plane classes; check COL_BRAIN / KEEP_CLASSES.');

fprintf('\nBrain-plane images used: %d\n', numel(files));
disp(table(categories(labels), countcats(labels), 'VariableNames', {'Class','Count'}));
disp(table(categories(machine), countcats(machine), 'VariableNames', {'US_Machine','Count'}));

%% ---------------- 2. Patient-disjoint split ----------------
trIdx = find(isTrain);  teIdx = find(~isTrain);
trPat = unique(patient(trIdx));
valPat = trPat(randperm(numel(trPat), round(0.10*numel(trPat))));
isVal  = ismember(patient(trIdx), valPat);
vaIdx  = trIdx(isVal);  trIdx = trIdx(~isVal);
assert(isempty(intersect(patient(trIdx), patient(teIdx))), 'Patient leakage between train and test');
fprintf('Train %d | Val %d | Test %d images\n', numel(trIdx), numel(vaIdx), numel(teIdx));

%% ---------------- 3-5. Train and evaluate (all machines) ----------------
[net, R] = train_and_eval(files, labels, machine, trIdx, vaIdx, teIdx, EPOCHS, BATCH, LR, OUT_DIR, 'all_machines');
save(fullfile(OUT_DIR, 'mobilenetv2_fetal_brain_planes.mat'), 'net', 'R');

%% ---------------- 6. Optional: leave-one-machine-out ----------------
if RUN_LOMO
    machines = categories(machine);
    LOMO = table('Size',[0 3],'VariableTypes',{'string','double','double'}, ...
                 'VariableNames',{'HeldOutMachine','TestImages','Accuracy'});
    for m = 1:numel(machines)
        held = machine == machines{m};
        tr_m = setdiff(find(~held & isTrain), vaIdx);
        te_m = find(held);
        if numel(te_m) < 30 || numel(unique(labels(tr_m))) < 3; continue; end
        [~, Rm] = train_and_eval(files, labels, machine, tr_m, vaIdx(~held(vaIdx)), te_m, ...
                                 EPOCHS, BATCH, LR, OUT_DIR, ['lomo_' matlab.lang.makeValidName(machines{m})]);
        LOMO = [LOMO; {string(machines{m}), numel(te_m), Rm.accuracy}]; %#ok<AGROW>
    end
    disp(LOMO); writetable(LOMO, fullfile(OUT_DIR, 'leave_one_machine_out.csv'));
end

%% =========================================================================
function [net, R] = train_and_eval(files, labels, machine, trIdx, vaIdx, teIdx, EPOCHS, BATCH, LR, OUT_DIR, tag)
    inSize = [224 224 3];
    aug = imageDataAugmenter('RandRotation',[-10 10], 'RandXTranslation',[-10 10], ...
                             'RandYTranslation',[-10 10], 'RandScale',[0.9 1.1]);
    mk = @(idx) imageDatastore(files(idx), 'Labels', labels(idx));
    dsTr = augmentedImageDatastore(inSize, mk(trIdx), 'DataAugmentation', aug, 'ColorPreprocessing','gray2rgb');
    dsVa = augmentedImageDatastore(inSize, mk(vaIdx), 'ColorPreprocessing','gray2rgb');
    dsTe = augmentedImageDatastore(inSize, mk(teIdx), 'ColorPreprocessing','gray2rgb');

    % MobileNetV2 with a new 3-class head; class weights counter imbalance
    lg = layerGraph(mobilenetv2);
    cls = categories(labels);
    cnt = countcats(labels(trIdx));
    w = (sum(cnt) ./ (numel(cnt) * cnt))';
    lg = replaceLayer(lg, 'Logits', fullyConnectedLayer(numel(cls), 'Name','Logits', ...
                      'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10));
    lg = replaceLayer(lg, 'ClassificationLayer_Logits', ...
                      classificationLayer('Name','output','Classes',cls,'ClassWeights',w));

    opts = trainingOptions('adam', 'InitialLearnRate',LR, 'MaxEpochs',EPOCHS, ...
        'MiniBatchSize',BATCH, 'Shuffle','every-epoch', 'ValidationData',dsVa, ...
        'ValidationFrequency',50, 'ValidationPatience',5, 'L2Regularization',1e-4, ...
        'OutputNetwork','best-validation-loss', 'Plots','training-progress', 'Verbose',true);
    net = trainNetwork(dsTr, lg, opts);

    [pred, scores] = classify(net, dsTe, 'MiniBatchSize', BATCH);
    yt = labels(teIdx);
    C  = confusionmat(yt, pred, 'Order', categorical(cls));
    tp = diag(C); prec = tp ./ max(sum(C,1)',1); rec = tp ./ max(sum(C,2),1);
    f1 = 2*prec.*rec ./ max(prec+rec, eps);
    R.accuracy = mean(pred == yt);
    R.macro_precision = mean(prec); R.macro_recall = mean(rec); R.macro_f1 = mean(f1);
    R.confusion = C; R.classes = cls; R.scores = scores;
    fprintf('\n[%s] Test accuracy %.4f | macro P %.4f R %.4f F1 %.4f\n', tag, R.accuracy, ...
            R.macro_precision, R.macro_recall, R.macro_f1);

    % Per-class table
    perClass = table(string(cls), prec, rec, f1, 'VariableNames', {'Class','Precision','Recall','F1'});
    disp(perClass); writetable(perClass, fullfile(OUT_DIR, [tag '_per_class.csv']));

    % Per-ultrasound-machine accuracy (device robustness, G1/G7)
    mt = machine(teIdx); mc = categories(removecats(mt));
    acc = arrayfun(@(k) mean(pred(mt == mc{k}) == yt(mt == mc{k})), (1:numel(mc))');
    n   = arrayfun(@(k) nnz(mt == mc{k}), (1:numel(mc))');
    perMachine = table(string(mc), n, acc, 'VariableNames', {'US_Machine','TestImages','Accuracy'});
    disp(perMachine); writetable(perMachine, fullfile(OUT_DIR, [tag '_per_machine.csv']));
    R.per_machine = perMachine;

    % Confusion-matrix figure
    f = figure('Visible','off');
    confusionchart(yt, pred, 'Normalization','row-normalized', ...
        'Title', sprintf('MobileNetV2 on FETAL\\_PLANES\\_DB brain planes (%s)', strrep(tag,'_','\_')));
    exportgraphics(f, fullfile(OUT_DIR, [tag '_confusion.png']), 'Resolution', 300); close(f);
end
