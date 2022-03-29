#pkgs <- names(sessionInfo()$otherPkgs)
pkgs_list <- list("mlr3verse" , "mlr3"    ,   "data.table", "forcats"   , "stringr"   , "dplyr"  ,    "purrr"  ,    "readr"  ,    "tidyr"   ,   "tibble"    , "ggplot2"  ,  "tidyverse" , "skimr" )
install.packages(pkgs_list)
library("skimr")
library("tidyverse")
library("data.table")
library("mlr3verse")
#View(patients)
 

  
patients <- readr::read_csv("heart_failure.csv")
skim(patients)
 

  
patients$fatal_mi <- as.factor(patients$fatal_mi)

png("explore_bar.png")
DataExplorer::plot_bar(patients, ncol = 3)
dev.off()

png("explore_hist.png")
DataExplorer::plot_histogram(patients, ncol = 3)
dev.off()


# data for heart patients
png("explore_mi.png")
DataExplorer::plot_boxplot(patients, by = "fatal_mi", ncol = 3)
dev.off()

set.seed(100)


heart_task <- TaskClassif$new(id = "heart", 
                              backend = patients,
                              target = "fatal_mi",
                              positive = "1")

# sets value train test split ratio to 0.8, 1/5 (1/folds) ratio
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)


lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")


design = benchmark_grid(
  tasks = list(heart_task),
  learners = list(lrn_baseline,lrn_log_reg,lrn_ranger,lrn_xgboost),
  resamplings = list(cv5)
)

bmr = benchmark(design)

 

  
names <- list("Featureless","Logistic Regression","Random Forest", "XGBoost")

png("misclass_all.png")
autoplot(bmr) + ggplot2::scale_x_discrete(labels = names) + ggtitle("Misclassification Error") + ylim(0,0.5) + ylab("Value")
dev.off()

png("acc_all.png")
autoplot(bmr, measure = msr("classif.auc")) + ggplot2::scale_x_discrete(labels = names) + ggtitle("Accuracy") + ylim(0.3,1) + ylab("Value")
dev.off()

png("roc_all.png")
autoplot(bmr, type = "roc") + ggplot2::scale_color_hue(labels = names)
dev.off()

png("prc_all.png")
autoplot(bmr, type = "prc") + ggplot2::scale_color_hue(labels = names)
dev.off()

out <- bmr$aggregate(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr")))
# 
# acc <- out$classif.acc
# auc <- out$classif.auc

scores <- c("Mean Misclassification Error","Accuracy","Area Under Curve","False Positive Rate", "False Negative Rate")

out2 <- data.frame(out$classif.ce,out$classif.acc,out$classif.auc, out$classif.fpr, out$classif.fnr,row.names = names)

colnames(out2) <- scores

melted <- melt(out2)
melted$learner <- rep(names,5)

png("bar_all.png",width = 1000)
ggplot(melted, aes(fill=as.character(learner), y = value, x = variable)) +
  geom_bar(position="dodge", stat="identity") +
  ggtitle("Cross Validated Scores") +
  xlab("Metric") + ylab("Value") +
  labs(fill = "Learner")
dev.off()

png("scores_table.png", height = 40*nrow(out2), width = 175*ncol(out2))
gridExtra::grid.table(out2)
dev.off()


  
rf <- resample(heart_task, lrn_ranger, cv5, store_models = TRUE)

png("rf_roc.png")
autoplot(rf,"roc")
dev.off()

png("rf_prc.png")
autoplot(rf,"prc")
dev.off()

print(rf$aggregate(msr("classif.acc")))

 
  
lrn_ranger$param_set
 

  
set.seed(100)

# highest so far, 9, 9, 20, 9

n=9

search_space = ps(
  mtry = p_dbl(lower = 1, upper = n),
  min.node.size = p_dbl(lower = 1, upper = n)
)

measure = msr("classif.acc")

evals20 = trm("evals", n_evals = 20)

tuner = tnr("grid_search", resolution = n)

instance = TuningInstanceSingleCrit$new(
  task = heart_task,
  learner = lrn_ranger,
  resampling = cv5,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)

at = AutoTuner$new(
  learner = lrn_ranger,
  resampling = rsmp("cv", folds = 5),
  measure = measure,
  search_space = search_space,
  terminator = evals20,
  tuner = tuner
)
#at

tuner$optimize(instance)

 
  

outer_resampling = rsmp("cv", folds = 3)

rr = resample(heart_task, at, outer_resampling, store_models = TRUE)

 
  
#  mtry = 3, min.node.size = 7, acc = 0.875
results <- extract_inner_tuning_results(rr)
#rr$aggregate()
#rr$score()
results <- data.frame(results$iteration,results$mtry,results$min.node.size,results$classif.acc)
#rowname(results) <- NULL
png("tuning_acc.png", height = 40*nrow(results), width = 125*ncol(results))
gridExtra::grid.table(results)
dev.off()
 

  
# feature selection trial

terminator = trm("evals", n_evals = 10)
fselector = fs("random_search")

at = AutoFSelector$new(
  learner = lrn_ranger,
  resampling = rsmp("cv", folds = 5),
  measure = measure,
  terminator = terminator,
  fselector = fselector
)

grid = benchmark_grid(
  task = heart_task,
  learner = list(at, lrn_ranger),
  resampling = rsmp("cv", folds = 3)
)


bmr2 = benchmark(grid, store_models = TRUE)
bmr2$aggregate(msr("classif.acc"))


Accuracy <- bmr2$aggregate(msr("classif.acc"))$classif.acc

sc <- data.frame(Accuracy,row.names = list("fselector","All Features"))
png("fsel_acc.png", height = 50*nrow(sc), width = 200*ncol(sc))
gridExtra::grid.table(sc)
dev.off()

png("bar_acc.png")
barplot(c(84.27,87.5),names.arg = c("Base Prediction Model","Tuned Prediction Model"), ylim=c(0,100),xlab = "Models",ylab = "Percentage", main="Accuracies", col=rgb(0.8,0.1,0.1,0.6))
dev.off()

 