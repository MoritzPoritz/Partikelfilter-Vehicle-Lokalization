library(psych)
boxplot(dummy_data$imu, dummy_data$lidar, names= c("Mean RMSE IMU", "Mean RMSE LIDAR"))
wilcox.test(dummy_data$imu, dummy_data$lidar, paired= TRUE, exact=FALSE, correct = TRUE, conf.int = TRUE, alternative = "greater")
wilcox.test(dummy_data$imu, dummy_data$lidar, paired= TRUE, exact=FALSE, correct = TRUE, conf.int = TRUE, alternative= "less")


