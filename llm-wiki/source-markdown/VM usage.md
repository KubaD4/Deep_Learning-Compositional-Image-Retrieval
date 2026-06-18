Azure Virtual Machine with a GPU

Tutorial: how to login & use it

1. Email

You should receive an email with
the invite to join the lab. Click the
blue button.

2. Login

You will be redirected to this login
page.
Use your @studenti.unitn.it
account! (See next slide)

2. Login

Use your @studenti.unitn.it
account!
Simply enter your email and hit the
blue button.

Note: only the student who
received the email can do this!

2. Login

After hitting the blue button in the
previous page, you will be
redirected to the usual UniTN login
page.
Log in using your data, and hit the
red button.

2. Login

You will be redirected to Azure’s
login page. You can choose to
remain connected to the account
by hitting “Yes” (blue button).

3. Dashboard

You are ﬁnally in the dashboard.
You should see something like this.
This is your virtual machine, and
you can see how many hours you’ve
used from your budget.

Note: remember to always turn
oﬀ the VM when you are done
using it, otherwise you will
waste hours!

4. Start the VM

Click the button on the bottom-
left side of the virtual machine’s
card. That will boot the VM.

Note: it might takes a while to
do that! Be patient and don’t
refresh the page.

4. Start the VM

Once the machine is booted, the
slider will say “In esecuzione” or
“Running”. The VM is ready.

5. Set a password

You can log into the VM via SSH.
However, you need to set a
password. You can do so by clicking
the circled icon. It will open the
popup shown here.

5. Set a password

Just choose a password of your
choice. You can always reset it. You
can share it with your teammates,
so that they can also log into the
VM.
Hit the blue button when you’ve
set the password.

Note: if the VM is oﬀ, you can’t
log into it. So the “admin” will
have to turn on the VM for the
teammates, too.

5. Set a password

Wait a couple of minutes so that
the password is correctly set.
Don’t refresh the page.

5. Set a password

Once it’s done, you should see
something like this. The initial
setup is completed, and you can
now log in into the VM.

6. Login

If you click again the icon in the red
circle, it should now show you the
address of your machine. It
provides you the command to
connect via SSH. Copy-paste it.

6. Login

Move to your terminal
and paste the command.
Then, hit Enter.

6. Login

If something like this happens, just type in “yes” and hit Enter.

6. Login

Type in your password (the one you set earlier in Azure’s web
console) and hit Enter.

Note: the password will not appear, and you will not see any
blinking indicator (as it usually happens in the terminal). Just
type the password and hit Enter, your keyboard is working!

6. Login

If you did everything correctly, you should
see something like this. You have eventually
made it to the VM!
You are an admin to this VM, so you can do
(almost) whatever you want with it.

You can test the GPU with `nvidia-smi`,
which will show you some info about the
GPU.

6. Login

`nvidia-smi` shows
something like this:

7. Environment setup

Now you need to install Python and PyTorch.
Using uv will make this easy.

Just type in the following commands:

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
mkdir <your_project_folder>
cd <your_project_folder>
uv init --python 3.12
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

7. Environment setup

This is the output that you should expect
from the commands in the previous slide.

8. Test the installation

Now you can test the GPU
(and if PyTorch recognizes it)
by running Python in
interactive mode, as shown
here.
No errors: everything is
working ﬁne.

Turn oﬀ
the VM!

Just a ﬁnal reminder to turn oﬀ the VM
when you’re not using it, otherwise you will
run out of budget pretty quickly.

