#' ---
#' title: "Euclidean Norm and Bland-Altman Plot"
#' output: html_document
#' ---
#' 
## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
library(dplyr)
library(ggplot2)
library(tidyr)
library(glue)
library(this.path)

#path_root <- '/media/Data03/Thesis/code/Inference/inference/Coordinates'
file.path()
path_root <- dirname(this.path())
path_figures <- file.path(path_root,'Figures')
path_tables <- file.path(path_root,'Tables')
path_tables <- file.path('/media/Data03/Projects/Paper_Method_Auto_Semi_Manual/tables')

#path_japan <- '/media/Data03/Projects/Paper_Method_Auot_Semi_Manual/figures'
path_japan <- file.path('/media/Data03/Projects/Paper_Method_Auto_Semi_Manual/figures')

#path_japan <- '/media/Data03/Projects/Automated_Electrode_Extraction/Poster_Japan/Poster_Japan_3/images'

#path_figures <- '/home/spencer/Dokumente/Models_R/Models_R/Figures'
tempdir()

#' ## Code style
#' ### read in for LMM
## -----------------------------------------------------------------------------

#path <- "/home/spencer/Dokumente/Models_R/Models_R/Tables"
df <- read.csv(file.path(path_root, "corrected_electrode_positions.csv"))
View(df)
# 'sub-002', 'sub-012', 'sub-004', 'sub-005' were measured again in different group, therefore outlier

df <- subset(df, select = -c(Unnamed..0, date))

df <- df %>%
  group_by(Subject) %>%
  mutate(Experiment = case_when(
    Experiment == "YourExperimentName" ~ first(Experiment[startsWith(Experiment, "P")]),
    TRUE ~ Experiment
  )) %>%
  ungroup()

print(unique(df$Experiment))
# original
# delete all control data points but do not delete if you want check precision
#df <- df[df$Session != "ses-0", ]
df$Method[df$run == "baseline"] <- "baseline"
df$Method[df$Method == "half-automated"] <- "semi-automated"
View(df)



## -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------
create_bland_altman_plot_all <- function(df, path_japan, path_figures, method_1,method_2,COI) {

  # Determine file name and title
  title <- "All Electrodes and Areas"
  
  # Define a color-blind friendly palette for 3 levels
  cbf_colors <- c("#E69F00", "#009E73", "#56B4E9", "#CC79A7")
  cbf_shapes <- c(25, 24, 23)

  if (grepl("Euclidean_Norm",COI, fixed = TRUE))  {
    sd_multiplier <- 1.96
    central_line <- mean(df$Difference)  # mean difference +- 1.96 SD
    LOA1 <- mean(df$Difference) + 1.96 * sd(df$Difference)
    LOA2 <- mean(df$Difference) - 1.96 * sd(df$Difference)
 } else {
sd_multiplier <- 1.5
    central_line <- median(df$Difference)
    LOA1 <- median(df$Difference) + 1.5 * IQR(df$Difference)
    LOA2 <- median(df$Difference) - 1.5 * IQR(df$Difference)
}

      if (COI == "Euclidean_Norm")
        {x_text <- "Euclidean Norm"}
      else
        {x_text <- COI}
  
  # Create the Bland-Altman plot
  plot <- ggplot(df, aes(x = Mean, y = Difference, fill = Electrode, shape = Area)) +
    geom_point(alpha = 0.7, size = 7, color = "black") +
    geom_hline(yintercept = central_line, color = "blue", linetype = "dashed", size = 2.5) +
    geom_hline(yintercept = LOA1, color = "red", linetype = "dashed", size = 2.5) +
    geom_hline(yintercept = LOA2, color = "red", linetype = "dashed", size = 2.5) +
    labs(
      #title = title,
      x = glue("Mean of {x_text}"),
      y = glue("Difference in mm ({method_1} - {method_2})"),
      fill = "Electrode",  # Legend title for fill color
      shape = "Area"  # Legend title for shape
    ) +
    theme_minimal() +
    #coord_cartesian(expand = FALSE) + 
    theme(
      panel.background = element_rect(fill = "white", color = "black"),
      plot.background = element_rect(fill = "white"),
      axis.text = element_text(size = 20),
      axis.title = element_text(size = 22),
      #plot.title = element_text(size = 41, face = "bold"),
      legend.text = element_text(size = 20),
      legend.title = element_text(size = 22),
      legend.key.size = unit(5, "lines"),
      legend.position = "right",  # Main legend position
      legend.box = "vertical",  # Change to vertical box
      legend.direction = "vertical",  # Keep legend items vertical
      legend.box.just = "left",  # Align the legend box to the left
      #plot.margin = margin(0.1, 0.1, 0.1, 0.1)  # Adjust margins as needed 
    ) +
    scale_y_continuous(breaks = seq(10, -20, -5)) +
    scale_fill_manual(values = cbf_colors) + # Use the color-blind friendly palette for fill
    scale_shape_manual(values = cbf_shapes) + # Use different shapes for each electrode
    guides(
      fill = guide_legend(override.aes = list(shape = 21, size = 8, stroke = 0)),
      shape = guide_legend(override.aes = list(size = 8, stroke = 2))
    ) #+
    # Use the color-blind friendly palette
    #scale_x_continuous(limits = c(70, 130), breaks = seq(70, 130, by = 20)) +
    #scale_y_continuous(limits = c(-35, 25), breaks = seq(-35, 25, by = 5)) # Set x and y axis limits # You can change the color palette as needed

    if (COI == "Euclidean_Norm")
    { plot <-plot +
    scale_x_continuous(limits = c(70, 130), breaks = seq(70, 130, by = 20)) +
    scale_y_continuous(limits = c(-35, 10), breaks = seq(-35, 10, by = 5)) # Set x and y axis limits # You can change the color palette as needed
    }
    else {
           { plot <-plot +
    scale_x_continuous(limits = c(-100, 100), breaks = seq(-100, 100, by = 25)) +
    scale_y_continuous(limits = c(-55, 55), breaks = seq(-55, 55, by = 10)) # Set x and y axis limits # You can change the color palette as needed
    }
    }

  # Determine file name
  file_name <- glue("BlandAltmanPlot_{method_1}_{method_2}_{COI}.png")

  # Save the plot
  ggsave(file.path(path_japan, file_name), plot = plot, width = 10, height =10, dpi = 300)
  ggsave(file.path(path_figures, file_name), plot = plot, width = 10, height = 10, dpi = 300)
  

     # Calculate the parameters
  bias <- mean(df$Difference)
  LoA1 <- bias + 1.96 * sd(df$Difference)
  LoA2 <- bias - 1.96 * sd(df$Difference)

  # Count points within LoA
  points_within_loa <- sum(df$Difference >= LoA2 & df$Difference <= LoA1)
  points_outside_loa <- nrow(df) - points_within_loa
  total_points <- nrow(df)
  percentage_within_loa <- (points_within_loa / total_points) * 100

      # Count points outside LoA belonging to rDLPFC
  points_outside_rdlpfc <- sum((df$Difference < LoA2 | df$Difference > LoA1) & df$Area == "rDLPFC")
  result <- df[df$Difference < LoA2 | df$Difference > LoA1, c("Subject", "Session", "run","Electrode",method_1,method_2)]
  # Then, save the result to a CSV file
  write.csv(result, file = file.path(path_tables,glue("points_outside_LoA_{method_1}_{method_2}_{COI}.csv")), row.names = FALSE)
  write.csv(result, file = file.path(path_japan,"points_outside_LoA__{method_1}_{method_2}_{COI}.csv"), row.names = FALSE)
  
  # Create filename using glue (ensure {method_1}, {method_2}, and {COI} exist in your environment)
    txt_file <- file.path(path_tables, glue::glue("points_outside_LoA_{method_1}_{method_2}_{COI}.txt"))

    # Redirect console output to file
    sink(txt_file)
    # Print the information for points outside LoA
  cat("Points outside Limits of Agreement:\n")
print(result)
cat("\nBias:", round(bias, 4), 
    "\nUpper Limit of Agreement (LoA1):", round(LoA1, 4),
    "\nLower Limit of Agreement (LoA2):", round(LoA2, 4),
    "\nPoints within LoA:", points_within_loa, "out of", total_points,
    "\nPercentage within LoA:", round(percentage_within_loa, 2), "%",
    "\nPoints outside LoA:", points_outside_loa,
    "\nPoints outside LoA belonging to rDLPFC:", points_outside_rdlpfc, "\n")

    # Return output to console
    sink()

  return(plot)
}


## -----------------------------------------------------------------------------
create_bland_altman_plot <- function(df, path_japan, path_figures, path_tables,method_1,method_2,COI, electrode = NULL) {

  
  # Apply filter if electrode is provided
  if (!is.null(electrode)) {
    df <- df %>% filter(Electrode == electrode)
  }
  
  # Determine file name and title
  title <- if (!is.null(electrode)) {
    glue("{electrode}")
  } else {
    "All Electrodes"
  }
  # Define a color-blind friendly palette for 3 levels
  cbf_colors <- c("#E69F00", "#56B4E9", "#009E73")
  # Create the Bland-Altman plot

    if (grepl("Euclidean_Norm",COI, fixed = TRUE))  {
    sd_multiplier <- 1.96
    central_line <- mean(df$Difference)  # mean difference +- 1.96 SD
    LOA1 <- mean(df$Difference) + 1.96 * sd(df$Difference)
    LOA2 <- mean(df$Difference) - 1.96 * sd(df$Difference)
 } else {
sd_multiplier <- 1.5
    central_line <- median(df$Difference)
    LOA1 <- median(df$Difference) + 1.5 * IQR(df$Difference)
    LOA2 <- median(df$Difference) - 1.5 * IQR(df$Difference)
}

  plot <- ggplot(df, aes_string(x = "Mean", y = "Difference", color = "Electrode")) +
    geom_point(alpha = 0.7, size = 5) +
    geom_hline(yintercept = central_line, color = "blue", linetype = "dashed",size = 2.5) +
    geom_hline(yintercept = LOA1, color = "red", linetype = "dashed",size = 2.5) +
    geom_hline(yintercept = LOA2, color = "red", linetype = "dashed",size = 2.5) +
    labs(
      title = title,
      x = glue("Mean of {COI}"),
      y = glue("Difference ({method_1} - {method_2})"),
      color = "Area"
    ) +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white", color = "black"),
      plot.background = element_rect(fill = "white"),
      axis.text = element_text(size = 35),
      axis.title = element_text(size = 31),
      plot.title = element_text(size = 41, face = "bold"),
      legend.text = element_text(size = 35),
      legend.title = element_text(size = 38),
      legend.key.size = unit(5, "lines"), 
      legend.position = "top",
      legend.direction = "horizontal",
    ) +
    scale_color_manual(values = cbf_colors)    #+  # Use the color-blind friendly palette
    #scale_x_continuous(limits = c(70, 130), breaks = seq(70, 130, by = 10)) +
    #scale_y_continuous(limits = c(-25, 15), breaks = seq(-25, 15, by = 5)) # Set x and y axis limits # You can change the color palette as needed


  # Determine file name
  file_name <- if (!is.null(electrode)) {
    glue("BlandAltmanPlot_{electrode}_{method_1}_{method_2}_{COI}.png")
  } else {
    "BlandAltmanPlot_{method_1}_{method_2}_{COI}.png"
  }

   file_csv <- if (!is.null(electrode)) {
    glue("BlandAltmanPlot_{electrode}_{method_1}_{method_2}_{COI}.csv")
  } else {
    "BlandAltmanPlot_{method_1}_{method_2}_{COI}.csv"
  }

  write.csv(df, file = file.path(path_tables,file_csv), row.names = FALSE)
  # Save the plot
  ggsave(file.path(path_japan, file_name), plot = plot, width = 10, height = 8, dpi = 300)
  ggsave(file.path(path_figures, file_name), plot = plot, width = 10, height = 8, dpi = 300)
  
  return(plot)
}

## -----------------------------------------------------------------------------
create_bland_altman_plot_Area <- function(df, path_japan, path_figures, path_tables,method_1,method_2,COI, electrode = NULL, Area = NULL) {

  
  # Apply filter if electrode is provided
  if (!is.null(electrode)) {
    df <- df %>% filter(Electrode == electrode)
  }

    # Apply filter if electrode is provided
  if (!is.null(area)) {
    df <- df %>% filter(Area == area)
  }
  
  # Determine file name and title
 title <- if (!is.null(electrode) & is.null(area)) {
    electrode
  } else if (!is.null(electrode) & !is.null(area)) {
    glue("{electrode}_{area}")
  } else {
    "All Electrodes"
  }
  # Define a color-blind friendly palette for 3 levels
  cbf_colors <- c("#E69F00", "#56B4E9", "#009E73")
  # Create the Bland-Altman plot

    if (grepl("Euclidean_Norm",COI, fixed = TRUE))  {
    sd_multiplier <- 1.96
    central_line <- mean(df$Difference)  # mean difference +- 1.96 SD
    LOA1 <- mean(df$Difference) + 1.96 * sd(df$Difference)
    LOA2 <- mean(df$Difference) - 1.96 * sd(df$Difference)
 } else {
sd_multiplier <- 1.5
    central_line <- median(df$Difference)
    LOA1 <- median(df$Difference) + 1.5 * IQR(df$Difference)
    LOA2 <- median(df$Difference) - 1.5 * IQR(df$Difference)
}

  plot <- ggplot(df, aes_string(x = "Mean", y = "Difference", color = "Area")) +
    geom_point(alpha = 0.7, size = 5) +
    geom_hline(yintercept = central_line, color = "blue", linetype = "dashed",size = 2.5) +
    geom_hline(yintercept = LOA1, color = "red", linetype = "dashed",size = 2.5) +
    geom_hline(yintercept = LOA2, color = "red", linetype = "dashed",size = 2.5) +
    labs(
      title = title,
      x = glue("Mean of {COI} Experimen:{AOI}"),
      y = glue("Difference ({method_1} - {method_2})"),
      color = "Area"
    ) +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white", color = "black"),
      plot.background = element_rect(fill = "white"),
      axis.text = element_text(size = 31),
      axis.title = element_text(size = 28),
      plot.title = element_text(size = 35, face = "bold"),
      legend.text = element_text(size = 31),
      legend.title = element_text(size = 33),
      legend.key.size = unit(5, "lines"), 
      legend.position = "top",
      legend.direction = "horizontal",
    ) +
    scale_color_manual(values = cbf_colors)    #+  # Use the color-blind friendly palette
    #scale_x_continuous(limits = c(70, 130), breaks = seq(70, 130, by = 10)) +
    #scale_y_continuous(limits = c(-25, 15), breaks = seq(-25, 15, by = 5)) # Set x and y axis limits # You can change the color palette as needed


  # Determine file name
  file_name <- if (!is.null(electrode) & is.null(area)) {
    glue("BlandAltmanPlot_{electrode}_{method_1}_{method_2}_{COI}_{AOI}.png")
  } else if (!is.null(electrode) & !is.null(area)) {
    glue("BlandAltmanPlot_{electrode}_{area}_{method_1}_{method_2}_{COI}_{AOI}.png")
  } 
  else {
    "BlandAltmanPlot_{method_1}_{method_2}_{COI}_{AOI}.png"
  }

   file_csv <- if (!is.null(electrode)) {
    glue("BlandAltmanPlot_{electrode}_{method_1}_{method_2}_{COI}_{AOI}.csv")
  } else {
    "BlandAltmanPlot_{method_1}_{method_2}_{COI}_{AOI}.csv"
  }

  write.csv(df, file = file.path(path_tables, file_csv), row.names = FALSE)

  # Save the plot
  ggsave(file.path(path_japan, file_name), plot = plot, width = 10, height = 8, dpi = 300)
  ggsave(file.path(path_figures, file_name), plot = plot, width = 10, height = 8, dpi = 300)
  
  return(plot)
}


run_all_input <- function(df,method_1, method_2, COI,AOI, path_root, path_japan, path_figures,path_tables){

  if (grepl("full-automated",method_1, fixed = TRUE) |
  grepl("full-automated",method_2, fixed = TRUE)) {
      rater_1 <- "Network"
      rater_2 <- "Kira"
  } else {
  rater_1 <- "Kira"
  rater_2 <- "Kira"
  }


  filtered_data <- df %>%
      dplyr::filter(Method %in% c(method_1, method_2) & Rater %in% c(rater_1,rater_2))



  View(filtered_data)
  write.csv(filtered_data, glue("filtered_corrected_electrode_positions_{method_1}_{method_2}.csv"), row.names = FALSE)

  #' 
#' 
## -----------------------------------------------------------------------------
filtered_data <- filtered_data %>%
  group_by(Subject, Session, run, Experiment, Electrode, Rater, Method) %>%
  summarise(
    X = mean(Coordinates[Dimension == "X"], na.rm = TRUE),
    Y = mean(Coordinates[Dimension == "Y"], na.rm = TRUE),
    Z = mean(Coordinates[Dimension == "Z"], na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(Euclidean_Norm = sqrt(X^2 + Y^2 + Z^2))
View(filtered_data)

#' 
## -----------------------------------------------------------------------------


# Assuming your dataset is named 'df'
filtered_data <- filtered_data %>%
  dplyr::mutate(Area = case_when(
    Experiment == "P1" ~ "rOTC",
    Experiment == "P3" ~ "lTPJ",
    Experiment == "P7" ~ "rDLPFC",
    TRUE ~ NA_character_  # This will set NA for any other values
  ))
 View(filtered_data) 

#' 
   # Extract baseline data (ses-0 and baseline runs)
  baseline_data <- filtered_data %>%
    filter(Method == "baseline") %>%
    select(Subject, Session, run, Area, Electrode, Method, !!sym(COI))
  
  # Extract non-baseline data (all other methods)
  non_baseline_data <- filtered_data %>%
    filter(Method != "baseline") %>%
    select(Subject, Session, run, Area, Electrode, Method, !!sym(COI))
  
  # Create a template of all unique sessions and runs from non_baseline_data
  session_run_template <- non_baseline_data %>%
    distinct(Subject, Session, run, Area, Electrode)
  
  # Repeat baseline data for all sessions and runs
  repeated_baseline_data <- session_run_template %>%
    left_join(baseline_data, by = c("Subject", "Area", "Electrode")) %>%
    select(-Session.y, -run.y) %>%  # Remove the original ses-0 and baseline columns
    rename(Session = Session.x, run = run.x)  # Rename columns back to Session and run
  
  # Combine the repeated baseline data with the non-baseline data
  combined_data <- bind_rows(non_baseline_data, repeated_baseline_data)

  View(combined_data)
## -----------------------------------------------------------------------------
# Prepare data for Bland-Altman plot
bland_altman_data <- combined_data %>%
  select(Subject, Session, run, Area, Electrode, Method, !!sym(COI)) %>%
  pivot_wider(names_from = Method, values_from = !!sym(COI)) %>%
  filter(!is.na(!!sym(method_1)) & !is.na(!!sym(method_2)))

# Calculate the differences and means for the Bland-Altman plot
bland_altman_data <- bland_altman_data %>%
  mutate(
    Difference = !!sym(method_1) - !!sym(method_2),
    Mean = (!!sym(method_1) + !!sym(method_2)) / 2
  )

#View(bland_altman_data)

# is you need more detailed plots uncomment this function
#create_bland_altman_plot(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Anode')
#create_bland_altman_plot(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Cathode1')
#create_bland_altman_plot(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Cathode2')
#create_bland_altman_plot(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Cathode3')

#create_bland_altman_plot_Area(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Anode',AOI)
#create_bland_altman_plot_Area(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Cathode1',AOI)
#create_bland_altman_plot_Area(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Cathode2',AOI)
#create_bland_altman_plot_Area(bland_altman_data, path_japan, path_figures, path_tables,method_1,method_2,COI,'Cathode3',AOI)

#' 
# Call the function
create_bland_altman_plot_all(bland_altman_data, path_japan, path_figures,method_1,method_2,COI)

return(bland_altman_data)
}

## -loop thour all possible differences and method pairs
diff_list <- c("Euclidean_Norm", "X", "Y", "Z")
#method_list <- c("baseline","full-automated", "semi-automated", "manually")
method_list <- c("full-automated", "semi-automated", "manually")
method_pairs <- list(
  c("full-automated", "semi-automated"),
  c("full-automated", "manually"),
  c("semi-automated", "manually")
)

#method_pairs <- list(
#  c("baseline", "manually"),
#  c("baseline", "semi-automated"),
#  c("baseline", "full-automated")
#)
Area_List <- c('rOTC','lTPJ','rDLPFC')

for (element in diff_list){ 
  COI <- element
  print(COI)
  for (pair in method_pairs){
    method_1 <- pair[1]
    method_2 <- pair[2]
    print(paste("Method 1:", method_1))
    print(paste("Method 2:", method_2))
    print("---")  # Add a separator for clarity


    #run_all_input(df,method_1, method_2, COI, path_root, path_japan, path_figures,path_tables)
    
    for (area in Area_List){
        AOI <- area
     #if filter of subject, session, run for first method pair "full-automated", "semi-automated" exists use it    
    df_1 <- df
    if (exists("filter_df")){
    # filter for methode comparison  
     df_1 <- df_1 %>% semi_join(filter_df, by = c("Subject", "Session", "run"))
     write.csv(df_1, glue("filtered_corrected_electrode_positions_{method_1}_{method_2}_{COI}_{AOI}.csv"), row.names = FALSE)
    # filter for baseline
    #  df_1 <- df_1 %>% semi_join(filter_df, by = c("Subject"))
    }

    filter_df <- run_all_input(df_1,method_1, method_2, COI, AOI, path_root, path_japan, path_figures,path_tables)
    #run_all_input(df,method_1, method_2, COI, AOI, path_root, path_japan, path_figures,path_tables)
 
    
    }
}
}
