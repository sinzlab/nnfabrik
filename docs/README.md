## How to contribute
To contribute to the documentations, simply clone the repo and create new (or edit existing) `rst` files in this folder.

To check the changes as you are making them there are two services, in the `docker-compose.yml`, that should run at the same time:

* `docs`: builds the html files and makes the content available on port 3000
* `sphinx`: updates the content every 30 seconds

Here are the steps to run these two services and see the changes (almost) as you are making them:

1. in the shell, navigate to the repo directory

2. run the `docs` service (in a screen session) via

   ```shell
   docker-compose up docs
   ```

3. run the `sphinx` service (in a separate screen session) via

   ````shell
   docker-compose up sphinx
   ````

4. you can now access the documentation in your browser at `localhost:3000`. If you are running this on a server, then you need a port mapping between your local machine and port 3000 on the server, and then you can access the documentation on the port you mapped to on your local machine

