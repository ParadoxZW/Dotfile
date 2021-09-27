# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/shaozw/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/shaozw/minconda3/etc/profile.d/conda.sh" ]; then
        . "/home/shaozw/minconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/shaozw/minconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export TERM=xterm-256color
source ~/antigen.zsh
# DEFAULT_USER=shaozw

# 加载oh-my-zsh库
antigen use oh-my-zsh

# 加载原版oh-my-zsh中的功能(robbyrussell's oh-my-zsh).
antigen bundle git
antigen bundle heroku
antigen bundle pip
antigen bundle lein
antigen bundle command-not-found

# 语法高亮功能
antigen bundle zsh-users/zsh-syntax-highlighting

# 代码提示功能
antigen bundle zsh-users/zsh-autosuggestions

# 自动补全功能
antigen bundle zsh-users/zsh-completions

# 加载主题
antigen theme robbyrussell/oh-my-zsh themes/agnoster
# antigen theme robbyrussell/oh-my-zsh themes/avit
# antigen theme robbyrussell/apple

# 保存更改
antigen apply

source ~/.zsh_profile
