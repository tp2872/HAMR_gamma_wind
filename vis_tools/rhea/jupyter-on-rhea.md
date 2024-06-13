Setting up Jupyter on Rhea
==========================

The following procedure can be used to run Jupyter server instances on Rhea
while allowing connections to them from a local (i.e. a laptop) browser.

This is a band-aid procedure in leiu of dedicated infrastructure for spinning up
Jupyter instances at the OLCF. Ideally we would offer a host running a
JupyterHub service where users could spin up secured, private Jupyter servers
that offload work transparently to dynamically started ipycluster backend jobs
through the batch system.

If you like the above idea, please mention it in the OLCF User Survey. With enough
voices of support, we could push to aquire the necessary infrastructure.

Meanwhile
---------

The Jupyter compute kernels should be run on *reserved batch nodes*. That is to
say **not** on the shared login nodes. Compute/resource intensive processes on
the login nodes can be killed without warning. The web browser used to access
the notebook interface is best run on your local machine, so the connection to
the notebook server must be tunneled out of the compute nodes.

## Installation

The [jupyter-on-rhea.pbs](jupyter-on-rhea.pbs) batch script in this repo
launches a Jupyter server on a single batch node and sets up a script to create
the necessary SSH tunnel to access it. In order to work, you will need to have
Jupyter installed somewhere in your `$PYTHONPATH`. You can arrange for this 
by using the standard module

```
module load python/2.7.15-anaconda2-2018.12
```

## Secure the Server

The connection to the server is not encrypted nor password protected by
default. As anyone can SSH to any node on Rhea, it is possible for other users
to connect to your notebook server and then **generate and run code as your
user if left unsecured!**

To harden the server, create a skeleton configuration file and **set an access
password**:

```bash
$ jupyter notebook --generate-config
$ python -c "from IPython.lib import passwd; p=passwd(); print '''c.NotebookApp.password = u'%s' ''' % p"
Enter password: 
Verify password: 
c.NotebookApp.password = 'sha1:123:some:456:secret:789:password:012:hash:3456789abcd'
```

**Manually insert** the hashed/salted password line that is output (similar to given above) to your Jupyter profile
config file (typically `$HOME/.jupyter/jupyter_notebook_config.py`).

Now that access is password protected, you must **encrypt all communication
traffic**, otherwise someone can simply intercept the unencrypted stream and
read your data and passwords. To enable encryption, generate a self-signed x509
server certificate/key pair. The DN data used is mostly arbitrary, but you must
use `127.0.0.1` for the hostname/server name as modern browsers will reject the
certificate if it does not match the URL that will be used to access the
notebook server:

```bash
$ openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout ~/.jupyter/mykey.key -out ~/.jupyter/mycert.pem
$ chmod go= ~/.jupyter/mykey.key
```

and set the options in `$HOME/.jupyter/jupyter_notebook_config.py`; in the below two lines please replace `$USER` with your user name:

```python
c.NotebookApp.certfile = '/ccs/home/$USER/.jupyter/mycert.pem'
c.NotebookApp.keyfile = '/ccs/home/$USER/.jupyter/mykey.key'
```

When using TLS encryption, you must explicitly use `https` instead of `http` in
the URL used to access the server. **NOTE:** Chrome will refuse to support the `https`
connection; to avoid dealing with this you can use e.g. Firefox.


### Running the Server

Starting the server is done by launching the PBS script with appropriate PBS
options:

```bash
$ qsub jupyter-on-rhea.pbs
```

The job will place an executable at `$HOME/.jupyter_connect` which contains
instructions on how to attach to the server. To see the instrucdtions, type
```
cat $HOME/.jupyter_connect
```

Typically, you must issue a
variation of

```bash
ssh -f -L 127.0.0.1:8080:127.0.0.1:XXXXXX $USER@rhea.ccs.ornl.gov /ccs/home/$USER/.jupyter_connect
```

on your local workstation and direct your local browser to `http://127.0.0.1:8080`. 

## MPI Capabilities and Ipyparallel

Setup as per the above instructions, the kernels all run on a single node. It is
possible to extend this setup to use the ipython cluster *ipcluster* backend and
the `$PBS_NODEFILE` to allow the kernel to run parallel tasks. See the notebook
[interactive_notebooks_with_mpi_on_rhea.ipynb](interactive_notebooks_with_mpi_on_rhea.ipynb)
in this repo for instructions on setting up an ipycluster backend.

## Server Uptime

The server is killed at least every 48 hours so you will want to **make sure
your work is saved often**. You can add a line like:

```bash
qsub -W depend=afternotok:$PBS_JOBID ok_jupyter.pbs
```

near the top of the `jupyter-on-rhea.pbs` batch script to resubmit the job
automatically to keep a server up, but you will still need to re-establish the
tunnel each time it goes down. 

## Allocation Consumption

This does consume your Rhea allocation so just keeping the server up and not
using it to crunch numbers is wasteful. It is perhaps the best practice to
do interactive development work on a local jupyter instance and then run a
dedicated python script in a batch job to make the most efficient use of your
allocation.

## Custom Re-configuration

Any of the configuration details should be tuned to your needs.
Specifically, the ports may need to be different for your case. You may want to
change `c.NotebookManager.notebook_dir` to use a different path than the default
so as to keep your toplevel `$HOME` directory tidy.

## Troubleshooting

Issues with this approach that could be fixed but have not yet been addressed
include:

1. **Cannot connect through local `CLIENT_PORT`**
  * Check that a server job is actually running with `showq -u $USER`.
  * Check that you are using the `https` prefix in your URL if using TLS encryption.
  * Check that you only have one tunnel open on the local machine. The tunnels
    are supposed to close when the connection is broken, but if they hang open,
    new tunnels will be assigned different ports than `CLIENT_PORT`.  There is a
    fix for this, but I have not documented it...

1. **Login password is rejected even though it is entered correctly.**
   * This is a symptom of multiple users attempting to access different servers
     through the same login node and port number. The connection will succeed
     but the password will be rejected because the server you are accessing is
     someone else's. Even though the login screen will look the same, you can
     use the TLS certificate information in your browser to verify if you are
     interacting with your server (ie, it is using your certificate) or someone
     else's. If this happens, use a different *random* LOGIN_PORT in the batch
     script.

1. **Cannot start a new server jobs even though no server job is running.**
   * New servers won't start while a `$HOME/.jupyter_connect` script exists.
     This script is used as a lockfile, but is sometimes not removed when the
     last server job exits. Verify that no server job is running using `showq -u $USER`
     and if none is running, simply delete `$HOME/.jupyter_connect` to
     allow a new server job to start. This is a symptom of the crude method of
     creating and clearing lockfiles used by this technique and should be
     improved.

