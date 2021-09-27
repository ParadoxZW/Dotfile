" #############################################
" Author: Canopus - zw.shao@foxmail.com
" Description: minimize plugin, heavily optimized,
" easy to understand and easy to use.
" Last Modified: 2021/9/27
" #############################################

" use `:verbose map <key>` to check key bindings.

" for debug
" py import sys, random;print(sys.version);print(random.__file__)
" autocmd BufWritePost $MYVIMRC source $MYVIMRC " eval vimrc immediately 

"""""""""""""""""""""""""""""""""""""""""""
" 1. Basic Config fot Editing
"""""""""""""""""""""""""""""""""""""""""""
set nocompatible  " must be the first line
set backspace=indent,eol,start " force <BS> work properly
set showcmd " show command in output buffer

" enable filetype check
filetype plugin indent on

" set autoindent in following 4 lines
set tabstop=4
set expandtab
set softtabstop=4
set sw=4  " shiftwidth

" config fold
set foldenable
set foldmethod=indent
set foldcolumn=0
set foldlevel=200
set foldclose=all
" use <space> to toggle fold
nnoremap <space> @=((foldclosed('.') < 0) ? 'zc' : 'zo')<cr>

" customize key mapping
" nnoremap <leader>a <C-a> " inc
" nnoremap <leader>x <C-x> " dec

" Window Switching
nnoremap <leader>a <c-w>h
nnoremap <leader>d <c-w>l
nnoremap <leader>s <c-w>j
nnoremap <leader>w <c-w>k
" inoremap <esc>h <esc><c-w>h
" inoremap <esc>l <esc><c-w>l
" inoremap <esc>j <esc><c-w>j
" inoremap <esc>k <esc><c-w>k

" let mapleader = '\' "it's default

" eval buffer
nnoremap <leader>e :%y\|@"<cr>


"""""""""""""""""""""""""""""""""""""""""""
" 2. Look and Appearance
"""""""""""""""""""""""""""""""""""""""""""
" Always show statusline
set laststatus=2
" Use 256 colours
set t_Co=256
" Config a beautiful Statusline, after `> pip install powerline`
set rtp+=/home/shaozw/miniconda3/envs/py2/lib/python2.7/site-packages/powerline/bindings/vim/
" Theme
set background=dark
colorscheme solarized

set nu
set cursorline

"""""""""""""""""""""""""""""""""""""""""""
" 3. Plugin Management
" INFO: Vim-plug is needed. Installation can 
" be done by:
" curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
"   https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
"
" Every time add a new plugin in vimrc, run
" :PlugInstall
"""""""""""""""""""""""""""""""""""""""""""
call plug#begin('~/.vim/plugged')
Plug 'neoclide/coc.nvim', {'branch': 'release'}
call plug#end()


"""""""""""""""""""""""""""""""""""""""""""
" 4. Config COC
" INFO: After COC itself is installed, run
" :CocInstall coc-pyright
" :CocInstall coc-tabnine
"""""""""""""""""""""""""""""""""""""""""""
" Always show the signcolumn, otherwise it would shift the text each time
" diagnostics appear/become resolved.
if has("patch-8.1.1564")
  " Recently vim can merge signcolumn and number column into one
  set signcolumn=number
else
  set signcolumn=yes
endif

" press Enter to choose top recommended item
inoremap <silent><expr> <cr> pumvisible() ? coc#_select_confirm()
                              \: "\<C-g>u\<CR>\<c-r>=coc#on_enter()\<CR>"


"""""""""""""""""""""""""""""""""""""""""""
" 5. My Implementation of Simple Tools
"""""""""""""""""""""""""""""""""""""""""""

"*********EasyComment**************************
func! EasyComment()
    let line_str = getline('.')
    let c_symbol = '#'
    if index(['vim'], &filetype) != -1
        let c_symbol = '"'
    elseif index(['c', 'cpp', 'java', 'javascript'], &filetype) != -1
        let c_symbol = '//'
    elseif index(['python', 'perl', 'bash'], &filetype) != -1
        let c_symbol = '#'
    elseif index(['lisp'], &filetype) != -1
        let c_symbol = ';'
    elseif index(['matlab', 'tex'], &filetype) != -1
        let c_symbol = '%'
    endif
    if line_str =~ ('^\s*' . c_symbol)
        exec "s/" . c_symbol . "\\s*//"
    else
        exec "normal! I" . c_symbol . "\<space>\<ESC>$"
    endif
endfunc

noremap <C-\> :call EasyComment()<cr>
"*********EasyComment**************************


"*********Auto Parenthesis*********************
func! Pair()
    if (&paste == 1)
        return 0 
    endif
    let c = getline('.')[col('.')-1]
    if c =~ '\s' || c == '' || c == ')' || c == ']' || c == '}'
        return 1
    else
        return 0
    endif
endfunc

inoremap <expr> " Pair() ? '""<left>' : '"'
" inoremap <expr> ' Pair() ? "''<left>" : "'"
inoremap <expr> ( Pair() ? '()<left>' : '('
inoremap <expr> [ Pair() ? '[]<left>' : '['
" inoremap <expr> { Pair() ? '{}<left>' : '{'
inoremap {<CR> {}<left><CR><CR><Up><TAB>
"*********Auto Parenthesis*********************


"*********Toggle Paste*************************
function! TogglePaste()
    if(&paste == 0)
        set paste
        echo "Paste Mode Enabled"
    else
        set nopaste
        echo "Paste Mode Disabled"
    endif
endfunction

map <leader>p :call TogglePaste()<cr>
"*********Toggle Paste*************************


