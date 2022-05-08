import os
import csv

base_dir = 'C:/Users/zhi.qiao/PycharmProjects/pythonProject/webpage/data'
effecient_idlist_path = os.path.join(base_dir, 'idlist')
dir_cobb = os.path.join(base_dir, 'cobb')
dir_cpr2dsag = os.path.join(base_dir, 'cpr2dsag')
dir_mpr2dsag = os.path.join(base_dir, 'mpr2dsag')
dir_ori3dct = os.path.join(base_dir, 'ori3dct')
dir_cpr3dct = os.path.join(base_dir, 'cpr3dct')

template_head = \
'''
<html>
<head>
    <link type="text/css" rel="stylesheet" href="../static/reset.css" />
    <title>Title</title>
    <style>
        div {position:relative}
        p{text-align: center;}
        table td{
        border-bottom:1px solid black;
        }
    </style>
</head>
<body>
'''

template_title = \
'''
<td colspan="6" align="center" style="font-size: 100">hello world</td>
'''

template_tail = \
'''
</body>
</html>
'''

def get_idlist(idlist_path):
    effecient_idlist = []
    with open(idlist_path,'r') as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            if idx > 0 and len(row[0])>0:
                effecient_idlist.append(row[0])
    return effecient_idlist

effecient_idlist = get_idlist(effecient_idlist_path)

print (effecient_idlist)



def create_row_script_update(pid):
    str_start = \
        '''
        <tr>
        '''
    str_end = \
        '''
        </tr>
        <td colspan="6" height="1" color="00cccc">
        '''
    str_col_patientID = '<td width="16%" style="border: 50px solid white; word-wrap:break-word;"><p style="font-size: 50">{0}</p></td>'.format(pid)
    str_col_ori3dct = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><div id="{0}_ori3dct"><img id="{1}_ori3dct_img" src={2} border=0 width="1000" height="1000"><span style="font-size: 50; position: absolute; top: 80; left: 80;"><div id="{3}_ori3dct_img_id">{4}</div></span></div></div><p style="font-size: 50">ori3dct</p></td>
    '''
    str_col_cpr3dct = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><div id="{0}_cpr3dct"><img id="{1}_cpr3dct_img" src={2} border=0 width="1000" height="1000"><span style="font-size: 50; position: absolute; top: 80; left: 80;"><div id="{3}_cpr3dct_img_id">{4}</div></span></div><p style="font-size: 50">cpr3dct</p></td>
    '''
    str_col_mpr2dsag = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><img id="{0}_mpr2dsag_img" src={1} border=0 width="1000" height="1000"><p style="font-size: 50">mpr2dsag</p></td>
    '''
    str_col_cpr2dsag = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><img id="{0}_cpr2dsag_img" src={1} border=0 width="1000" height="1000"><p style="font-size: 50">cpr2dsag</p></td>
    '''
    str_col_cobb = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><img id="{0}_cobb_img" src={1} border=0 width="1000" height="1000"><p style="font-size: 50">cobb</p></td>
    '''
    str_col_ori3dct = str_col_ori3dct.format(pid, pid, os.path.join(dir_ori3dct, pid, '1.jpg'), pid, 1)
    str_col_cpr3dct = str_col_cpr3dct.format(pid, pid, os.path.join(dir_cpr3dct, pid, '1.jpg'), pid, 1)
    str_col_mpr2dsag = str_col_mpr2dsag.format(pid, os.path.join(dir_mpr2dsag, pid+'.jpg'))
    str_col_cpr2dsag = str_col_cpr2dsag.format(pid, os.path.join(dir_cpr2dsag, pid+'.jpg'))
    str_col_cobb = str_col_cobb.format(pid, os.path.join(dir_cobb, pid+'.jpg'))

    return str_start \
            + str_col_patientID\
            + str_col_ori3dct \
            + str_col_cpr3dct \
            + str_col_mpr2dsag \
            + str_col_cpr2dsag \
            + str_col_cobb \
            +str_end

def create_row_script(pid):
    str_start = \
        '''
        <tr>
        '''
    str_end = \
        '''
        </tr>
        <td colspan="6" height="1" color="00cccc">
        '''
    str_col_patientID = '<td width="16%" style="border: 50px solid white; word-wrap:break-word;"><p style="font-size: 50">{0}</p></td>'.format(pid)
    str_col_ori3dct = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><div id="{0}_ori3dct"><img id="{1}_ori3dct_img" src={2} border=0 width="1000" height="1000"></div><p style="font-size: 50">ori3dct</p></td>
    '''
    str_col_cpr3dct = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><div id="{0}_cpr3dct"><img id="{1}_cpr3dct_img" src={2} border=0 width="1000" height="1000"></div><p style="font-size: 50">cpr3dct</p></td>
    '''
    str_col_mpr2dsag = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><img id="{0}_mpr2dsag_img" src={1} border=0 width="1000" height="1000"><p style="font-size: 50">mpr2dsag</p></td>
    '''
    str_col_cpr2dsag = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><img id="{0}_cpr2dsag_img" src={1} border=0 width="1000" height="1000"><p style="font-size: 50">cpr2dsag</p></td>
    '''
    str_col_cobb = \
    '''
    <td width="16%" style="border: 50px solid white; word-wrap:break-word;"><img id="{0}_cobb_img" src={1} border=0 width="1000" height="1000"><p style="font-size: 50">cobb</p></td>
    '''
    str_col_ori3dct = str_col_ori3dct.format(pid, pid, os.path.join(dir_ori3dct, pid, '1.jpg'))
    str_col_cpr3dct = str_col_cpr3dct.format(pid, pid, os.path.join(dir_cpr3dct, pid, '1.jpg'))
    str_col_mpr2dsag = str_col_mpr2dsag.format(pid, os.path.join(dir_mpr2dsag, pid+'.jpg'))
    str_col_cpr2dsag = str_col_cpr2dsag.format(pid, os.path.join(dir_cpr2dsag, pid+'.jpg'))
    str_col_cobb = str_col_cobb.format(pid, os.path.join(dir_cobb, pid+'.jpg'))

    return str_start \
            + str_col_patientID\
            + str_col_ori3dct \
            + str_col_cpr3dct \
            + str_col_mpr2dsag \
            + str_col_cpr2dsag \
            + str_col_cobb \
            +str_end

def createTable_html(effecient_idlist):
    str_start = \
    '''
    <table>
    '''
    str_end = \
    '''
    </table>
    '''
    str_middle = ''
    for pid in effecient_idlist:
        str_middle += create_row_script_update(pid)

    final_script = str_start + template_title + str_middle + str_end
    return final_script

def createTable_script(effecient_idlist):
    str_start = \
        '''
        <script type="text/javascript">
        '''
    str_end = \
        '''
        </script>
        '''
    control_template = \
    '''
    document.getElementById("%s").addEventListener("wheel", %s);
    var img=1;
    function %s(e) {
            var delta = Math.max(-1, Math.min(1, (e.wheelDelta || -e.detail)));
            if(delta > 0){
                img += 1
                if(img>%s) {img=1}
                }
            else{
                img -= 1
                if(img<1) {img=%s}
                }
            document.getElementById("%s").src = "%s/"+img+".jpg";
            document.getElementById("%s").innerHTML = img;
        }
    '''

    str_middle = ''
    for pid in effecient_idlist:
        local_control_ori3dct = control_template
        docuID = "%s_ori3dct"%pid
        funName = "func_"+docuID
        img_n = len(os.listdir(os.path.join(dir_ori3dct, pid)))
        imgID = "%s_ori3dct_img"%pid
        imgID_id = "%s_ori3dct_img_id" % pid
        imgpath = os.path.join(dir_ori3dct, pid).replace('\\', '/')
        str_middle += local_control_ori3dct%(docuID, funName, funName, img_n, img_n, imgID, imgpath, imgID_id)

        local_control_cpr3dct = control_template
        docuID = "%s_cpr3dct" % pid
        funName = "func_" + docuID
        img_n = len(os.listdir(os.path.join(dir_cpr3dct, pid)))
        imgID = "%s_cpr3dct_img" % pid
        imgID_id = "%s_cpr3dct_img_id" % pid
        imgpath = os.path.join(dir_cpr3dct, pid).replace('\\', '/')
        str_middle += local_control_cpr3dct % (docuID, funName, funName, img_n, img_n, imgID, imgpath, imgID_id)


    return str_start + str_middle + str_end

def creatHtml_page(tab_html, tab_script):
    return template_head + tab_html + tab_script + template_tail

def write_file(script):
    fw = open('./index.html','w')
    fw.write(script)
    fw.close()

tab_html = createTable_html(effecient_idlist)
tab_script = createTable_script(effecient_idlist)
Html_script = creatHtml_page(tab_html, tab_script)
print (Html_script)
write_file(Html_script)
