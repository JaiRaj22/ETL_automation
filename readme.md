# Enhanced Automated Data Analysis System

A Streamlit application for automated data cleaning, analysis, and visualization with machine learning algorithm suggestions.

## Features

- Automated data cleaning and preprocessing
- Intelligent ID column detection and removal
- Datetime detection and conversion
- Missing value handling
- Outlier detection and handling
- Categorical variable encoding
- Correlation analysis with visualizations
- PCA analysis
- Automatic plot generation based on data types
- Machine learning algorithm suggestions and evaluation

## Getting Started

### Prerequisites

- Docker installed on your system
- Git for version control

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-analysis-app.git
   cd data-analysis-app
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run index.py
   ```

4. Open your browser and go to `http://localhost:8501`

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t data-analysis-app .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 data-analysis-app
   ```

3. Open your browser and go to `http://localhost:8501`

## Deployment

This repository includes GitHub Actions workflows for automatic building and deploying of Docker images to GitHub Container Registry.

### Container Registry Access

To deploy the application:

1. Ensure you have the proper permissions to access the GitHub Container Registry
2. Pull the latest image:
   ```bash
   docker pull ghcr.io/yourusername/data-analysis-app:latest
   ```

3. Run the container:
   ```bash
   docker run -p 8501:8501 ghcr.io/yourusername/data-analysis-app:latest
   ```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.