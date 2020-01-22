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

easy_eval <- 
  fit_resamples(knn_wfl, 
                resamples =  cv_split, 
                control = tune::control_resamples(save_pred = TRUE))
easy_eval

tune::collect_predictions(easy_eval) %>%
  arrange(.row) %>%
  slice(1:5)
tune::collect_metrics(easy_eval)
tune::collect_metrics(easy_eval, summarize = FALSE)

penalty()
mixture()

glmn_param <- parameters(penalty(), mixture())

glmn_param

glmn_grid <- 
  grid_regular(glmn_param, levels = c(10, 5))

glmn_grid %>% slice(1:5)

set.seed(784652)
glmn_sfd <- grid_max_entropy(glmn_param, size = 50)
glmn_sfd %>% slice(1:5)

glmn_set <- parameters(lambda = penalty(), mixture())

# The ranges can also be set by their name:
glmn_set <- 
  update(glmn_set, lambda = penalty(c(-5, -1)))

mtry()

rf_set <- parameters(mtry(), trees())
rf_set

dials::finalize(rf_set, mtcars %>% dplyr::select(-mpg))

?dials::finalize
?parsnip::nearest_neighbor

dials::neighbors()
dials::weight_func()
dials::dist_power()

knn_param <- 
  tune::parameters(dials::neighbors(),
                    dials::weight_func())
knn_param

knn_mod <- parsnip::nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

knn_grid <- knn_param %>% grid_regular(levels = c(15, 5))

ctrl <- tune::control_grid(verbose = TRUE)

knn_tune <- tune::tune_grid(ames_rec, model = knn_mod,
                            resamples = cv_splits, grid = knn_grid,
                            control = ctrl)

knn_tune

tune::show_best(knn_tune, metric = "rmse", maximize = FALSE)

data("Chicago")

Chicago %>%
  slice()

library(stringr)
library(tidyverse)
library(tidymodels)
library(tune)
us_hol <- timeDate::listHolidays() %>%
  str_subset("(US)|(Ester)")

chi_rec <- 
  recipe(ridership ~., data = Chicago) %>%
  step_holiday(date, holidays = us_hol) %>%
  step_date(date) %>%
  step_rm(date) %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(one_of(!!stations)) %>%
  step_pca(one_of(!!stations), num_comp = tune())

chi_folds <- rolling_origin(Chicago, initial = 364 * 15, assess = 7 * 4, skip = 7 * 4, cumulative = FALSE)
chi_folds %>% nrow()
chi_folds
lm(ridership ~ ., Chicago)

glmn_sfd

glmn_rec <- 
  chi_rec %>%
  step_normalize(all_predictors())

glmn_mod <- 
  parsnip::linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")
glmn_mod

ctrl <- control_grid(save_pred = TRUE)

glmn_grid <- expand.grid(penalty = 10^seq(-3, -1, length = 20), mixture = (0:5)/5)

glmn_tune <- 
  tune::tune_grid(
    glmn_rec,
    model = glmn_mod,
    resamples = chi_folds,
    grid = glmn_grid,
    control = ctrl
  )

glmn_tune <-
  tune_grid(
    glmn_rec,
    model = glmn_mod,
    resamples = chi_folds,
    grid = glmn_grid,
    control = ctrl
  )
glmn_grid

tune::collect_metrics(glmn_tune)

tune::show_best(glmn_tune, metric = "rmse", maximize = FALSE)

best_glmn <- select_best(glmn_tune, metric = "rmse", maximize = FALSE)

best_glmn

glmn_pred <- tune::collect_predictions(glmn_tune)
glmn_pred

glmn_pred <- 
  glmn_pred %>%
  inner_join(best_glmn, by = c("penalty", "mixture"))

glmn_pred %>%
  ggplot(aes(x = .pred, y = ridership)) +
  geom_abline(col="red") +
  geom_point(alpha = 0.5) +
  coord_equal()

large_resid <- 
  glmn_pred %>%
  mutate(resid = ridership - .pred) %>%
  arrange(desc(abs(resid))) %>%
  slice(1:5)

library(lubridate)

Chicago %>%
  slice(large_resid$.row) %>%
  select(date) %>%
  mutate(day = wday(date, label = TRUE)) %>%
  bind_cols(large_resid)


glmn_rec_final <- prep(glmn_rec)
glmn_mod_final <- tune::finalize_model(glmn_mod, best_glmn)
glmn_mod_final

glmn_fit <- glmn_mod_final %>%
  fit(ridership ~., data = juice(glmn_rec_final))

glmn_fit

library(glmnet)

plot(glmn_fit$fit, xvar = "lambda")

library(vip)

vip(glmn_fit, num_features = 20L, 
    # Needs to know which coefficients to use
    lambda = best_glmn$penalty)
