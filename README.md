### Image Processing Server

## Installation
copy the folder to the Server. Point the terminal to the folder.
Please update the MACROS in ADTDefines.hpp file to the ukbench dataset and users folder location.

# Install NPM dependencies 
$ npm install

# Start local server
$ npm start


API - explanation
========================================

**adding user with email id**
curl -X POST localhost:3000/api/adduser_emailid -d "emailid=<email id of the user>"

**deleting the user with an email id**
curl -X POST localhost:3000/api/deletuser_emailid -d "emailid=<email id of the user>"

**add image to an user with email id**
curl -X POST localhost:3000/api/addimageuser_emailid --form "image=@<path of the image>" --form "emailid=<email id of the user>"

**delete an image from a user database**
curl -X POST localhost:3000/api/removeimageuser_emailid -d "emailid=<emailid of the user>" -d "imagename=<image name to be deleted>"

**retrieve a list of image ids given an image**
**Returns a comma seperated list of matching image names**
curl -X POST localhost:3000/api/retrieveimage_emailid --form "image=@<path of the image>" --form "emailid=<email id of the user>"

After starting local webserver, navigate to http://localhost:3000 where you should see similar page:

