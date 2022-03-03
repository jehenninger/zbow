Updated 220302 - Jonathan Henninger

Note: The "%" symbol below denotes a new line of the command prompt in the Terminal app, e.g.

    % print "Zebrafish are cool"

The "%" is the beginning of the command line. Typing in the rest should print "Zebrafish are cool" in the Terminal

## Downloading code:

1. Go to https://github.com/jehenninger/zbow
2. Click on "Code" and download the .zip file
3. Unzip the file and move the folder to a directory of your choosing (called "/path/to/zbow_code" here)

## Creating an isolated Python virtual environment that has all the packages you will need

1. Open the Terminal app
2. Enter the following to create a new virtual environment that is distinct from the default Python installation on the Mac. You can call it whatever you want, and the name is the last name in the path. "zbow-venv" might be appropriate. It can be in a different directory than the zbow_code.

    % python3 -m venv /path/to/zbow-venv
    
3. When installing packages in the virtual environment, you need to 'activate' it. So do this now:

    % source /path/to/zbow-venv/bin/activate
    
This should now place some text to the left of your username in parantheses, e.g. "(zbow-env)", to denote that you are in the virtual environment

4. Update "pip" which is the package installation software. The update is necessary because some packages don't install with previous pip versions:

    (zbow-venv) % pip install --upgrade pip
    
5. Install all necessary packages. I have included a "requirements.txt" file in the zbow_code that you can feed into pip to auto install everything:

    (zbow-venv) % pip install -r /path/to/zbow_code/requirements.txt
    
If you encounter any errors with installation, make sure your pip is upgraded (Step 4), and if that doesn't work, let me know!

6. You can now "deactivate" the environment as we are done setting it up:

    (zbow-venv) % deactivate
    
This should return you to a normal Terminal command prompt

## Running the Zbow app

1. The first thing you must do (only once) is to modify the "main.py" python text file in the zbow_code folder. At the top of the file, you should see:

    #!/Users/jonathanhenninger/zbow_venv/bin/python

Change this to point to your virtual environment. This line of text tells the script to use the virtual environment code to run everything.

    #!/path/to/zbow-venv/bin/python


2. In the Terminal app, change the directory to the zbow_code:

    % cd /path/to/zbow_code
    
3. You'll need to make the "main.py" file executable, otherwise you'll get a permissions error:

    % chmod +x ./main.py
    
The "chmod +x" modifies the permission of the file to allow it to run as an executable. The "./" signifies that "main.py" is in the current directory. This only needs to be done once.

4. Now you should be all set to run the program. From now on, all you need to do is open Terminal and type:

    % /path/to/zbow_code/main.py
    
    If you are in the starting directory
    
    OR
    
    % ./main.py
    
    If you are in the zbow_code directory
    