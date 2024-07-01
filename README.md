# Video Search and Indexing

## Overview
This project is a comprehensive solution for searching and indexing video content. It allows users to quickly find relevant videos and the exact timestamp within those videos, based on a short query video snippet.

The system works by pre-processing a database of videos to extract various visual and audio features, which are then used to generate a digital signature for each video. When a user provides a query video, the system analyzes it, creates a sub-signature, and matches it against the pre-computed signatures to find the best match.

The output of the system is an interactive media player that displays the matched video, starting at the frame corresponding to the query video.

## Key Features
- **Robust Video Indexing**: The system uses a combination of color, frequency coefficients, and sound features to create a unique digital signature for each video in the database.
- **Efficient Matching**: A vector database is used to store the pre-computed video signatures, allowing for fast and accurate matching of the query video sub-signature.
- **Interactive Media Player**: The final output is presented in a custom media player that allows users to play, pause, and reset the video, ensuring seamless playback of the matched content.
- **Extensible Architecture**: The modular design of the system makes it easy to incorporate additional feature extraction techniques or switch to different vector database solutions in the future.

## Architecture
The project is structured into the following modules:

- `preprocessing/`: Handles the extraction of visual and audio features from the videos and the generation of digital signatures which are then stored in the form of an embedding.
- `matching/`: Responsible for managing the vector database, performing the sub-signature matching, and identifying the best match.
- `player/`: Implements the custom media player for displaying the output.
- `utils/`: Provides utility functions and helper classes used across the project.
- `models/`: Defines the data models used throughout the project.

![video_search_engine drawio](https://github.com/maheshJosephSadashiv/com.dev.video-search-engine/assets/38533715/84f481bb-351c-4545-8b34-94083062fa40)
results:
BLUE: db video at the exact frame as query vid
ORANGE: query video
## Endpoint: 

/

    Method: GET, POST
    Description: This endpoint serves as the main entry point for the application. It renders either the index page or handles a POST request to search for a video file and display the results.

Request Parameters

    None

Request Body

    Method: POST
        Parameter: video (string, required)
            Description: The location of the video file to search for.

Responses

    200 OK
        Description: The index page is rendered, or the search results are displayed.
        Content: HTML content with the rendered page.
    404 Not Found
        Description: If the requested page or resource is not found.
        Content: Error page.
## Dependencies

- `brew reinstall ffmpeg`
- `pip install -r requirements.txt`
## Analysis
Computing the perceptual similarities with the query video and persisted video proved effective in searching. Here is a graph on the similarity index.
<img width="1300" alt="similarity_index" src="https://github.com/maheshJosephSadashiv/com.dev.video-search-engine/assets/38533715/693600a0-5efd-4288-a9c5-5007ac0e64df">

## Demo
<iframe width="560" height="315" src="https://youtu.be/yh6ETUFDFsM?si=wskHKFkw-q8EEyqn" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## License
This project is licensed under the [MIT License](LICENSE).
