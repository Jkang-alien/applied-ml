library(yardstick)

perf_metrics <- metric_set(ccc, rmse, rsq)

test_pred %>%
  perf_metrics(truth = log_price, estimate = .pred, na_rm = TRUE)

library(workflows)

ames_rec <- 
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) %>% 
  step_log(Sale_Price, base = 10) %>% 
  step_ns(Longitude, deg_free = tune("long df")) %>% 
  step_ns(Latitude,  deg_free = tune("lat df"))

ames_wfl <- 
  workflows::workflow() %>%
    add_recipe(ames_rec) %>%
    add_model(lm_mod)

ames_wfl

ames_wfl_fit <- fit(ames_wfl, ames_train)
ames_wfl_fit$fit
predict(ames_wfl_fit, ames_test %>% slice(1:5))

library(rsample)

set.seed(548623)

cv_split <- rsample::vfold_cv(ames_train)
cv_split

cv_split$splits[[1]] %>% analysis() %>% dim()
cv_split$splits[[1]] %>% assessment() %>% dim()

knn_mod <- 
  parsnip::nearest_neighbor(neighbors = 5) %>%
  set_engine("kknn") %>%
  set_mode("regression")

knn_wfl <- 
  workflows::workflow() %>%
  add_model(knn_mod) %>%
  add_formula(log10(Sale_Price) ~ Longitude + Latitude)

library(tune)

knn_mod <- 
  parsnip::nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

knn_wfl <- 
  workflows::workflow() %>%
  add_model(knn_mod) %>%
  add_recipe(ames_rec)

knn_param <- 
  knn_wfl %>%
  dials::parameters() %>%
  stats::update(
    'long df' = spline_degree(c(2,18)),
    'lat df' = spline_degree(c(2,18)),
    neighbors = neighbors(c(3,50)),
    weight_func = weight_func(values = c("rectangular", "inv", "gaussian", "triangular"))
  )
knn_param
