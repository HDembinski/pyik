# from http://code.activestate.com/recipes/576591-simple-shelve-with-linux-file-locking/
#
# Copyright (c) 21 Dec 2008 Michael Ihde
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import shelve
import fcntl
import types
from six.moves import builtins
from fcntl import LOCK_SH, LOCK_EX, LOCK_UN, LOCK_NB


def _close(self):
    shelve.Shelf.close(self)
    fcntl.flock(self.lckfile.fileno(), LOCK_UN)
    self.lckfile.close()


def open(filename, flag='c', protocol=None, writeback=False, block=True, lckfilename=None):
    """
    Open the sheve file, createing a lockfile at filename.lck.

    If block is False then a IOError will be raised if the lock cannot be acquired.
    """
    if lckfilename is None:
        lckfilename = filename + ".lck"
    lckfile = builtins.open(lckfilename, 'w')

    # Accquire the lock
    if flag == 'r':
        lockflags = LOCK_SH
    else:
        lockflags = LOCK_EX
    if not block:
        lockflags = LOCK_NB
    fcntl.flock(lckfile.fileno(), lockflags)

    shelf = shelve.open(filename, flag, protocol, writeback)
    shelf.close = types.MethodType(_close, shelf)
    shelf.lckfile = lckfile
    return shelf
