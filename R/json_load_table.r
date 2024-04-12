library(shiny)
library(jsonlite)
library(DT)

# Define UI
ui <- fluidPage(
  titlePanel("View JSON Data"),
  DTOutput("dataTable")
)

# Define server logic
server <- function(input, output) {
  output$dataTable <- renderDT({
    # Load JSON data
    # Replace the placeholder path with the actual path to your JSON file on your system
    jsonData <- fromJSON(subset_data.json, flatten = TRUE)
    
    # Convert the loaded JSON data to a dataframe for display
    dataTable <- as.data.frame(jsonData)
    
    # Use DT::datatable() for interactive table display
    DT::datatable(dataTable, options = list(pageLength = 10))
  })
}

# Run the application
shinyApp(ui = ui, server = server)
