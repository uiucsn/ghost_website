# GHOST Data Visualization Tool

This is an interactive tool for visualizing the host galaxies of supernovae in the Galaxies HOsting Supernovae and other Transients (GHOST) Catalog. It is based on code adapted from Matias Carrasco Kind's galaxy postage stamp viewer, which is written in Python and Javascript. The code reads the images from a single directory, and runs the asyncio server on the back end.

### Deployment

#### Simple deployment

Clone this repository:

		git clone https://github.com/uiucsn/ghost_website
		cd cutouts-explorer/python_server

Create a config file template:

		cp config_template.yaml config.yaml

Edit the `config.yaml` file to have the correct parameters, see [Configuration](#Configuration) for more info.

Start the server:

	   python3 server.py

Start the client and visit the url printed python_server:

	   python3 client.py

If you are running locally you can go to [http://localhost:8000/](http://localhost:8000/)

#### Docker

0. Create image from Dockerfile

        cd cutouts-explorer
        docker build -t cexp .

1. Create an internal network so server/client can talk through the internal network (is not need for now as we are exposing both services at the localhost)

        docker network create --driver bridge cutouts

2. Create local config file to be mounted inside the containers. Create `config.yaml` based on the template, and replace the image location.

3. Start the server container and attach the volume with images, connect to network and expose port 8888 to localhost

           docker run -d --name server -p 8888:8888 -v {PATH TO CONFIG FILE}:/home/explorer/server/config.yaml -v {PATH TO LOCAL IMAGES}:{PATH TO CONTAINER IMAGES} --network cutouts cexp python server.py

4. Start the client container, connect to network and expose the port 8000 to local host

           docker run -d --name client -p 8000:8000 -v {PATH TO CONFIG FILE}:/home/explorer/server/config.yaml  --network cutouts cexp python client.py

Now the containers can talk at the localhost. 
If you are running locally you can go to [http://localhost:8000/](http://localhost:8000/)

### Configuration

This is the template config file to use:

```
#### DISPLAY
display:
  dataname: '{FILL ME}' #Name for the sqlite DB and config file
  path: '{FILL ME}'
  nimages: 1200 #Number of objects to be displayed even if there are more in the folder
  xdim: 40 #X dimension for the display
  ydim: 30 #Y dimension for the display
  tileSize: 256 #Size of the tile for which images are resized at max zoom level
  minXrange: 0
  minYrange: 0
  deltaZoom: 3 #default == 3
#### SERVER
server:
  ssl: false #use ssl, need to have certificates
  sslName: test #prefix of .crt and .key files inside ssl/ folder e.g., ssl/{sslName.key}
  host: 'http://localhost' #if using ssl, change to https
  port: 8888
  rootUrl: '/cexp' #root url for server, e.g. request are made to /cexp/, if None use "/"
  #workers: None # None will default to the workers in the machine
#### CLIENT
client:
  host: 'http://localhost'
  port: 8000
#### OPERATIONS options
operation:
  updates: true #allows to update and/or remove classes to images, false and classes are fixed.
#### CLASSES
#### classes, use any classes from 0 to 9, class 0 is for hidden! class -1 is no class
classes:
    - Elliptical: 9
    - Spiral: 8
    - Other: 7
    - Delete: 0
```

Please email me (gaglian2@illinois.edu) if you have any questions/comments! 
