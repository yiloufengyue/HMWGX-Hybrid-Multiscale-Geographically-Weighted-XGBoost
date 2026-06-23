## 1. Load packages
library(tidyverse)
library(cowplot)
library(cols4all)
library(mgcv)
library(GWmodel)
library(broom)
library(spdep)
library(stringr)

set.seed(111)
size <- 50
total_points <- size * size

# 生成均匀分布的随机数
X1 <- runif(size*size, -1.5, 1.5)
X2 <- runif(size*size, -1.5, 1.5)
X3 <- runif(size*size, -1.5, 1.5)
X4 <- runif(size*size, -1.5, 1.5)
X5 <- runif(size*size, -1.5, 1.5)
X6 <- runif(size*size, -1.5, 1.5)

# 合并为矩阵
X <- cbind(X1, X2, X3, X4, X5, X6)

# 初始化矩阵
b6 <- matrix(0, nrow = size, ncol = size)
b3 <- matrix(0, nrow = size, ncol = size)
b4 <- matrix(0, nrow = size, ncol = size)

# 填充b6
for(i in 1:size) {
  for(j in 1:size) {
    b6[i, j] <- 2
  }
}

# 填充b4
for(i in 1:size) {
  for(j in 1:size) {
    b4[i, j] <- cos(pi * exp((i-1)/50)) * sin(pi * exp((j-1)/50)) * 3
  }
}

# 填充b3
for(i in 1:size) {
  for(j in 1:size) {
    b3[i, j] <- 3 * ((i-1 + j-1) / 99)
  }
}

# 生成坐标
u <- rep(0:(size-1), times = size)
v <- rep(0:(size-1), each = size)
coords <- cbind(u, v)

# 对角区域使用相同函数形式
f1 <- rep(0, length(X1))

# 创建区域掩码
center_u <- size / 2
center_v <- size / 2

# 划分四个象限
mask_top_left <- (u < center_u) & (v < center_v)      # 左上
mask_top_right <- (u >= center_u) & (v < center_v)    # 右上  
mask_bottom_left <- (u < center_u) & (v >= center_v)  # 左下
mask_bottom_right <- (u >= center_u) & (v >= center_v) # 右下

f1[mask_top_left] <- tanh(X1[mask_top_left] * pi)
f1[mask_bottom_right] <- sin(X1[mask_bottom_right] * pi)
f1[mask_top_right] <- abs(X1[mask_top_right]) * X1[mask_top_right]
f1[mask_bottom_left] <- X1[mask_bottom_left]^3

# f2
f2 <- rep(0, length(X2))
mask_left <- u < (size / 2)
mask_right <- !mask_left
f2[mask_left] <- abs(X2[mask_left]) * X2[mask_left]
f2[mask_right] <- tanh(X2[mask_right] * pi)

# f3, f4, f5, f6
f3 <- as.vector(b3) * as.vector(X3)
f4 <- as.vector(b4) * as.vector(X4)
f5 <- X5^3
f6 <- 2 * X6

# 合并所有f
fs <- cbind(f1, f2, f3, f4, f5, f6)

# 生成误差
err <- runif(size*size, -1.5, 1.5)

y_vector <- f1 + f2 + f3 + f4 + f5 + f6 + err

tryCatch({
  loaded_data <- data.frame(
    y = y_vector,
    x1 = X1,
    x2 = X2,
    x3 = X3,
    x4 = X4,
    x5 = X5,
    x6 = X6,
    lng = u,
    lat = v
  )
  
  required_cols <- c("y", "x1", "x2", "x3", "x4", "x5", "x6", "lng", "lat")
  if (!all(required_cols %in% colnames(loaded_data))) {
    stop("Internal error: Not all required columns were created in the synthetic data.")
  }
  
  cat("Dimensions of loaded_data:", dim(loaded_data)[1], "rows,", dim(loaded_data)[2], "columns.\n")
  cat("First few rows of loaded_data:\n")
  print(head(loaded_data))
  
}, error = function(e) {
  cat("Error during synthetic data structuring: ", e$message, "\n")
  stop("Synthetic data generation or structuring failed.")
})

# Rename columns and add necessary ones for the GAM model
model_data <- loaded_data %>%
  select(
    y = y,         # Response variable
    X1 = x1,       # Predictor 1
    X2 = x2,       # Predictor 2
    X3 = x3,       # Predictor 3
    X4 = x4,       # Predictor 4
    X5 = x5,       # Predictor 5
    X6 = x6,       # Predictor 6
    u = lng,       # Spatial coordinate 1 (mapped from lng)
    v = lat        # Spatial coordinate 2 (mapped from lat)
  ) %>%
  mutate(
    Intercept = 1 # Column of 1s for the spatially varying intercept
    # Add true_y and true_signal if they were in the CSV and you want them for comparison
    # true_y = true_y,
    # true_signal = true_signal
  )

# Get the number of locations from the loaded data
n_locations <- nrow(model_data)
cat("Number of data points loaded:", n_locations, "\n")

## 9. Fit the GGP-GAM model with SVCs for all predictors
# Model structure: y = beta0(u,v) + beta1(u,v)*X1 + ... + beta5(u,v)*X5 + error
gam_formula_svc_all <- as.formula(
  y ~ 0 +                      
    s(u, v, bs='gp', by=Intercept) + 
    s(u, v, bs='gp', by=X1) +       
    s(u, v, bs='gp', by=X2) +       
    s(u, v, bs='gp', by=X3) +       
    s(u, v, bs='gp', by=X4) +
    s(u, v, bs='gp', by=X5) +
    s(u, v, bs='gp', by=X6) 
)

cat("Fitting GGP-GAM model with SVCs for all predictors...\n")
gam_model_svc_all <- gam(gam_formula_svc_all, data = model_data, method = "REML")
cat("Model fitting complete.\n")

svc_vars <- c("Intercept", "X1", "X2", "X3", "X4", "X5", "X6")
# Prepare base newdata with all relevant predictor columns set to 0
base_newdata_preds_svc <- model_data %>%
  select(u, v, all_of(svc_vars)) %>% # Select coords + all SVC predictor columns
  mutate(across(all_of(svc_vars), ~ 0)) # Set all these columns to 0
# Add columns for estimated SVCs to the model_data dataframe
for (var_name in svc_vars) {
  cat("Extracting estimated SVC for:", var_name, "...\n")
  # Create specific newdata for this variable
  current_newdata <- base_newdata_preds_svc %>%
    mutate(!!sym(var_name) := 1) # Set the target variable to 1
  
  # Predict the SVC surface
  col_name <- paste0("estimated_beta_", var_name)
  model_data[[col_name]] <- predict(gam_model_svc_all, newdata = current_newdata, type = 'response')
}


# 2) Prediction values
model_data$predicted_y_svc_all <- predict(gam_model_svc_all, newdata = model_data, type = 'response')

## Save Location-based Results to CSV
# This includes coordinates, input variables, true y, true signal, estimated SVCs, and predicted y.
csv_output_filename <- "ggp_gam_svc_all_location_results.csv"
cat("\nSaving location-based results to:", csv_output_filename, "...\n")
write.csv(model_data, csv_output_filename, row.names = FALSE)
cat("Location-based results saved successfully.\n")

## Save Model Summary and Bandwidths to TXT
txt_output_filename <- "ggp_gam_svc_all_model_summary.txt"
cat("\nSaving model summary and bandwidths to:", txt_output_filename, "...\n")

# Use sink() to redirect output to the text file
sink(txt_output_filename)

cat("--- GGP-GAM (SVCs for All Predictors) Model Summary and Parameters ---\n\n")

cat("Model Formula:\n")
print(gam_formula_svc_all)
cat("\n")

cat("Full Model Summary:\n")
print(summary(gam_model_svc_all))
cat("\n")

cat("Smoothing Parameters (SP) for GP smooths (Bandwidths):\n")
# Get SP values and their names (smooth term labels)
sp_values <- gam_model_svc_all$sp
sp_labels <- sapply(gam_model_svc_all$smooth, function(x) x$label) # Labels like s(u,v):Intercept, s(u,v):X1 etc.
sp_df <- data.frame(Label = sp_labels, SP = sp_values)
print(sp_df)
cat("\n")


# Calculate and Print Overall Model Fit Metrics (Predicted y vs True y)
rsq <- function (x, y) cor(x, y) ^ 2
rmse <- function(x, y) sqrt(mean((x - y)^2))

cat("Overall Model Fit Metrics (Predicted y vs True y):\n")
cat("R-squared:", round(rsq(model_data$y, model_data$predicted_y_svc_all), 4), "\n")
cat("RMSE:", round(rmse(model_data$y, model_data$predicted_y_svc_all), 4), "\n")
cat("\n")

# --- Explanation for Non-linear Curves ---
cat("Regarding 'non-linear function relationship curves':\n")
cat("In this specific GGP-GAM model structure (using s(u,v, bs='gp', by=X_i) for all predictors),\n")
cat("the effect of each variable X_i is modeled as X_i * beta_i(u,v).\n")
cat("At any fixed location (u,v), this is a linear relationship with X_i, with the slope being beta_i(u,v).\n")
cat("The non-linearity in the model arises from the spatial variability of the coefficients beta_i(u,v),\n")
cat("which are 2D surfaces over the (u,v) coordinates, not 1D curves plotting the response vs X_i.\n")
cat("Therefore, standard 1D non-linear relationship curves (like partial effect plots for s(X_i) terms)\n")
cat("are not outputs of this model structure.\n")
cat("The estimated 'function relationship' for each variable X_i is its estimated spatial coefficient surface beta_i(u,v).\n")
 cat(paste0("These estimated SVC surfaces are saved as columns ('estimated_beta_", paste(svc_vars, collapse=", estimated_beta_"), "') in the CSV output file.\n"))
cat("\n")


cat("--- End of Model Summary ---\n")

# Stop redirecting output and close the file
sink()

cat("Model summary and bandwidths saved successfully to:", txt_output_filename, "\n")

#================================
#  10. Residual Bootstrap for 95% CI of spatial coefficients
#================================

cat("\nStarting residual bootstrap for coefficient confidence intervals...\n")

# ----------------------------
# Bootstrap settings
# ----------------------------
B <- 200   # number of bootstrap replications
coef_vars <- c("X1", "X2", "X3", "X4", "X5", "X6")
n <- nrow(model_data)

# ----------------------------
# Original fitted values and residuals
# ----------------------------
model_data$fitted_original <- predict(gam_model_svc_all, newdata = model_data, type = "response")
model_data$residual_original <- model_data$y - model_data$fitted_original

# ----------------------------
# Storage for bootstrap coefficient surfaces
# Each matrix: n_locations x B
# ----------------------------
bootstrap_betas <- list()
for (var in coef_vars) {
  bootstrap_betas[[var]] <- matrix(NA, nrow = n, ncol = B)
}

# ----------------------------
# Base newdata used to extract coefficient surfaces
# ----------------------------
svc_vars_extract <- c("Intercept", coef_vars)

base_newdata_preds_boot <- model_data %>%
  select(u, v, all_of(svc_vars_extract)) %>%
  mutate(across(all_of(svc_vars_extract), ~ 0))

# ----------------------------
# Bootstrap loop
# ----------------------------
for (b in 1:B) {
  cat("Bootstrap iteration:", b, "of", B, "\n")
  
  # 1) Shuffle residuals (residual bootstrap)
  boot_resid <- sample(model_data$residual_original, size = n, replace = FALSE)
  
  # 2) Generate bootstrap response
  y_boot <- model_data$fitted_original + boot_resid
  
  # 3) Create bootstrap dataset
  boot_data <- model_data
  boot_data$y <- y_boot
  
  # 4) Refit the same GAM-GGP model
  boot_fit <- tryCatch({
    gam(gam_formula_svc_all, data = boot_data, method = "REML")
  }, error = function(e) {
    cat("Bootstrap iteration", b, "failed:", e$message, "\n")
    return(NULL)
  })
  
  # If model fitting failed, skip this iteration
  if (is.null(boot_fit)) next
  
  # 5) Extract coefficient surfaces for X1 ~ X6
  for (var_name in coef_vars) {
    current_newdata <- base_newdata_preds_boot %>%
      mutate(!!sym(var_name) := 1)
    
    bootstrap_betas[[var_name]][, b] <- predict(
      boot_fit,
      newdata = current_newdata,
      type = "response"
    )
  }
}

cat("\nCalculating 95% confidence intervals...\n")

for (var_name in coef_vars) {
  beta_mat <- bootstrap_betas[[var_name]]
  
  # 有些bootstrap可能失败，去掉全NA列
  valid_cols <- colSums(!is.na(beta_mat)) > 0
  beta_mat_valid <- beta_mat[, valid_cols, drop = FALSE]
  
  cat("Valid bootstrap samples for", var_name, ":", ncol(beta_mat_valid), "\n")
  
  # Pointwise 95% CI at each location
  lower_ci <- apply(beta_mat_valid, 1, quantile, probs = 0.025, na.rm = TRUE)
  upper_ci <- apply(beta_mat_valid, 1, quantile, probs = 0.975, na.rm = TRUE)
  mean_boot <- apply(beta_mat_valid, 1, mean, na.rm = TRUE)
  sd_boot   <- apply(beta_mat_valid, 1, sd, na.rm = TRUE)
  
  # Save into model_data
  model_data[[paste0("estimated_beta_", var_name, "_boot_mean")]]  <- mean_boot
  model_data[[paste0("estimated_beta_", var_name, "_boot_sd")]]    <- sd_boot
  model_data[[paste0("estimated_beta_", var_name, "_ci_lower")]]   <- lower_ci
  model_data[[paste0("estimated_beta_", var_name, "_ci_upper")]]   <- upper_ci
}

bootstrap_ci_output_filename <- "ggp_gam_svc_all_bootstrap_ci_results.csv"
cat("\nSaving bootstrap CI results to:", bootstrap_ci_output_filename, "...\n")
write.csv(model_data, bootstrap_ci_output_filename, row.names = FALSE)
cat("Bootstrap CI results saved successfully.\n")
