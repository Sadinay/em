"""Non-destructive report-only collection for project03. Never copy models or dataset files."""
from pathlib import Path
from urllib.parse import unquote
import csv
import hashlib
import json
import os
import re
import shutil
ROOT=Path(__file__).resolve().parents[1]/'03_new_spmsm_project'
OUT=ROOT/'reports'
SP_NAME='SPα0.95-0.80-0.50_β0.01_掺杂率12.5pct_v1'
TEST_NAME='SPα0.80_β0.01_掺杂率12.5pct_v1'
ALLOWED={'.md','.txt','.png','.jpg','.jpeg','.svg','.html','.pdf','.csv','.json'}
SPECS=[
 ('SP0_掺杂率0pct_v3','V3冻结基线','SmallCNN V2、Mini-Inception V2、ResNet20 V2；VGG16四输入视图','0','0%',[('reports/V3','V3',True)]),
 ('SP0_掺杂率25pct_v1','普通热启动回放v1及6×20扩展','Polar90 VGG16；SmallCNN、Mini-Inception、ResNet20','0','25%',[('experiments/cnn_replay_update_v1/reports/vgg16','Polar90VGG16',True),('experiments/cnn_replay_update_v1/reports/logical6x20','6x20',True)]),
 ('SP0_掺杂率12.5pct_v2','普通热启动回放v2及6×20扩展','Polar90 VGG16；SmallCNN、Mini-Inception、ResNet20','0','12.5%',[('experiments/cnn_replay_update_v2/reports/vgg16','Polar90VGG16',True),('experiments/cnn_replay_update_v2/reports/logical6x20','6x20',True)]),
 (SP_NAME,'一次性SP＋回放首轮α筛选','Logical6x20SmallCNNV2；原生2×6×20','α=.95/.80/.50；β=.01','12.5%',[('experiments/cnn_shrink_perturb_v1/reports/small_cnn_v2','',True)]),
 ('SP不适用_掺杂率不适用_v01','历史FEMM映射和DeltaT诊断散件','FEMM；无CNN系数','不适用','不适用',[('reports','',False)]),
 ('SP不适用_掺杂率不适用_v02','历史拓扑数据整理报告','数据处理；无CNN系数','不适用','不适用',[('reports/spmsm_topology_dataset','',True)]),
 ('SP不适用_掺杂率不适用_v03','历史输入编码报告','输入表示；无CNN系数','不适用','不适用',[('reports/spmsm_inputs','',True)]),
 ('SP不适用_掺杂率不适用_v04','20260908角度设置审计','FEMM','不适用','不适用',[('reports/angle_settings_audit_20260908','',True)]),
 ('SP不适用_掺杂率不适用_v05','20260908初始内角29°审计','FEMM','不适用','不适用',[('reports/initial_angle_29_audit_20260908','',True)]),
 ('SP不适用_掺杂率不适用_v06','20260909跨设备pilot20核验','FEMM','不适用','不适用',[('femm_zone/workspaces/pilot20_cross_device_check_20260909','',False)]),
 ('SP不适用_掺杂率不适用_v07','20260913补样标注验收与冻结基线','补样G/F；冻结Polar90 VGG16特征','不适用','不适用',[('experiments/input_distribution_pilot_v1/post_femm_baseline_20260913','',False)]),
 ('SP不适用_掺杂率不适用_v08','20260908固定电流五基因核验','FEMM','不适用','不适用',[('femm_zone/results/fixed_current_5genes_20260908','',False)]),
 ('SP不适用_掺杂率不适用_v09','20260908初始电流相位核验','FEMM','不适用','不适用',[('femm_zone/results/initial_current_phase_20260908','',False)]),
 ('SP不适用_掺杂率不适用_v10','20260908电流模式五基因核验','FEMM','不适用','不适用',[('femm_zone/results/current_modes_5genes_20260908','',False)]),
 ('SP不适用_掺杂率不适用_v11','20260908MAT字面电流核验','FEMM','不适用','不适用',[('femm_zone/results/mat_literal_current_probe_20260908','',False)]),
]

# Stable project chronology, not report file modification dates (copies/regeneration change those).
CHRONOLOGY = {
 'SP不适用_掺杂率不适用_v01':('早期','00'),
 'SP不适用_掺杂率不适用_v02':('V3训练前','01'),
 'SP不适用_掺杂率不适用_v03':('V3训练前','02'),
 'SP0_掺杂率0pct_v3':('2026-09-03起，后续补齐','03'),
 'SP不适用_掺杂率不适用_v04':('2026-09-08','04'),
 'SP不适用_掺杂率不适用_v08':('2026-09-08','05'),
 'SP不适用_掺杂率不适用_v10':('2026-09-08','06'),
 'SP不适用_掺杂率不适用_v11':('2026-09-08','07'),
 'SP不适用_掺杂率不适用_v09':('2026-09-08','08'),
 'SP不适用_掺杂率不适用_v05':('2026-09-08','09'),
 'SP不适用_掺杂率不适用_v06':('2026-09-09','10'),
 'SP不适用_掺杂率不适用_v07':('2026-09-13','11'),
 'SP0_掺杂率25pct_v1':('2026-09-14；小模型扩展09-15','12'),
 'SP0_掺杂率12.5pct_v2':('2026-09-15','13'),
 SP_NAME:('2026-09-15，当前轮','14'),
}
# In the replay timeline, the historical V3 (40,000 old training samples) is v0.
legacy=[item for item in SPECS if item[0].startswith('SP不适用_')]
SPECS=[item for item in SPECS if not item[0].startswith('SP不适用_')]
SPECS=[('SP0_掺杂率0pct_v0','v0：40000旧数据冻结基线（历史V3）',*item[2:]) if item[0]=='SP0_掺杂率0pct_v3' else item for item in SPECS]
other_sources=[]
for item in legacy:
    for source,prefix,recursive in item[5]:
        other_sources.append((source,item[1]+('__'+prefix if prefix else ''),recursive))
SPECS.append(('other','其他历史图表和说明','FEMM核验、数据处理、输入表示等历史杂项','—','—',other_sources))
CHRONOLOGY['SP0_掺杂率0pct_v0']=CHRONOLOGY['SP0_掺杂率0pct_v3']
CHRONOLOGY['other']=('历史杂项','00')
SPECS.append((TEST_NAME,'固定模型测试：SP F-S与原始f0','SmallCNN V2；F-S SP α=.80，9500步；原始f0','α=.80；β=.01','12.5%',[('experiments/cnn_test_smallcnn_sp080_v1/report','',True)]))
CHRONOLOGY[TEST_NAME]=('2026-09-15，固定模型测试','15')
EXPANSION_NAME='SP0_掺杂率12.5pct_v3'
SPECS.append((EXPANSION_NAME,'f1特征覆盖扩样：10000基因FEMM任务包','f1 Polar90 VGG16；F-S，4000步无约束候选；本轮未训练CNN','0（选样模型）','12.5%（选样模型；本轮不训练）',[('experiments/input_distribution_expansion_f1_v1/report','',True)]))
CHRONOLOGY[EXPANSION_NAME]=('2026-09-16，f1覆盖扩样','16')
SPECS.sort(key=lambda item: CHRONOLOGY[item[0]][1])

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def eligible(p,source):
    if p.suffix.lower() not in ALLOWED or '__pycache__' in p.parts:return False
    if p.name in ('实验索引.md','实验索引.csv','归档清单.json','归档核验.json','README.md') and source==OUT:return False
    if any(x in p.name.lower() for x in ('prediction','checkpoint','labels_','waveforms_','targets_','output_checksums')):return False
    if source==OUT and p.suffix.lower()=='.json':return False
    # Outside dedicated report folders, take narrative and rendered figures only.
    if 'report' not in source.parts and 'reports' not in source.parts and p.suffix.lower() not in ('.md','.txt','.png','.svg','.html','.pdf'):return False
    return True


def main():
    mappings={};records=[];groups={};retained=[]
    for dirname,title,models,sp,ratio,sources in SPECS:
        dest=OUT/dirname;dest.mkdir(exist_ok=True)
        copied=[]
        for relative,prefix,recursive in sources:
            source=ROOT/relative
            if not source.exists():continue
            for p in sorted(source.rglob('*') if recursive else source.iterdir()):
                if not p.is_file() or not eligible(p,source):continue
                name='__'.join(([prefix] if prefix else [])+list(p.relative_to(source).parts))
                target=dest/name
                assert target.resolve().is_relative_to(dest.resolve()) and p.resolve()!=target.resolve()
                assert target not in mappings.values(),'Archive name collision'
                mappings[p.resolve()]=target;copied.append((p,target))
        groups[dirname]=(title,models,sp,ratio,copied)
    pattern=re.compile(r'(!?\[[^\]\n]*\]\()([^\n]*?)(\))')
    broken=[]
    for original,target in mappings.items():
        before=sha(original)
        if original.suffix.lower()=='.md':
            text=original.read_text(encoding='utf-8-sig')
            def rewrite(m):
                value=m.group(2).strip();url=value.strip('<>')
                if re.match(r'^(https?://|mailto:|#|data:)',url):return m.group(0)
                baseurl,sep,anchor=url.partition('#')
                source=(original.parent/unquote(baseurl)).resolve()
                converted=mappings.get(source,source)
                if not source.exists():
                    broken.append({'document':str(original.relative_to(ROOT)),'existing_link':url})
                rel=Path(os.path.relpath(converted,target.parent)).as_posix()
                return m.group(1)+'<'+rel+(sep+anchor if sep else '')+'>'+m.group(3)
            target.write_text(pattern.sub(rewrite,text),encoding='utf8')
        else:
            shutil.copy2(original,target)
        assert before==sha(original),'Source report was modified'
        records.append({'source':str(original.relative_to(ROOT)),'destination':str(target.relative_to(ROOT)),'source_sha256':before,'archive_sha256':sha(target),'markdown_links_rebased':original.suffix.lower()=='.md'})
    index=['# 03实验报告索引','','报告集中在本目录的规范子文件夹；每次实验一个文件夹，仅用SP、掺杂率和轮次命名。模型信息见本表和报告正文。','',
        'SP0表示普通初始化/热启动，无SP混合；掺杂率指更新batch中的新样本比例。v0对应40000旧数据训练的冻结基线（原历史V3）；非训练历史图表统一放入other。','',
        '原报告全部保留，旧名称目录是历史原件；新规范副本位于下表入口。只复制报告/摘要/图表，未搬动checkpoint或数据。历史报告可能包含当时已公布的测试指标，归档只复制原件，不用于本轮SP选择；SP系数筛选阶段保持测试集封存；之后经用户授权对固定模型及原f0进行了测试，见最新测试条目。','',
        '按实验进行时间从早到晚排列，最新在最下面。同日早期核验未逐一保存精确起止时间，按已知工作顺序排列，不用报告修改时间冒充实验时间。','', '| 时间 | 实验 | SP | 新样本掺杂率 | 网络/对象 | 报告入口 |','|---|---|---|---|---|---|']
    indexrows=[]
    for dirname,(title,models,sp,ratio,copied) in groups.items():
        dest=OUT/dirname
        lines=[f'# {title}','','模型/对象：'+models,'','SP：'+sp+'；新样本掺杂率：'+ratio+'。','',
            '原始结果保留。此处仅归档报告、摘要和图表；源代码、checkpoint及数据留在原目录。Markdown链接已调整，原始产物哈希见归档清单。v0对应历史V3的40000旧数据基线；历史原件中的V3字样及模型路径保留。','','## 报告入口','']
        docs=[p for _,p in copied if p.suffix.lower()=='.md']
        if not docs:lines+=['报告尚未生成；实验完成后更新。','']
        for p in docs:lines.append(f'- [{p.stem}](<{p.name}>)')
        lines+=['','## 图片','','图片在相应报告中直接嵌入；也可打开本文件夹的PNG查看。','']
        (dest/'README.md').write_text('\n'.join(lines),encoding='utf8')
        index.append(f'| {CHRONOLOGY[dirname][0]} | {title} | {sp} | {ratio} | {models} | [查看](<{dirname}/README.md>) |')
        indexrows.append({'time':CHRONOLOGY[dirname][0],'order':CHRONOLOGY[dirname][1],'experiment':title,'folder':dirname,'models':models,'sp':sp,'new_fraction':ratio,'report_files':len(copied),'source_folders':'; '.join(s[0] for spec in SPECS if spec[0]==dirname for s in spec[5])})
    (OUT/'实验索引.md').write_text('\n'.join(index)+'\n',encoding='utf8')
    with (OUT/'实验索引.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(indexrows[0]));writer.writeheader();writer.writerows(indexrows)
    (OUT/'README.md').write_text('# 03报告入口\n\n[打开实验索引](实验索引.md)\n\n40000旧数据冻结基线在补样时间线中记为v0（历史名称V3保留在原件/模型来源记录）。其他历史图表统一在 `other/`。新实验报告按 `SP值_掺杂率值_v轮次` 放入独立目录；文件夹不写模型名称，模型写入索引及报告正文。保留旧报告原件，不能以归档为由修改模型或数据。\n\n可从仓库根目录执行 `python ./maintenance/collect_03_reports.py` 更新已登记实验的报告副本和索引；新增实验在该脚本的SPECS表登记。\n',encoding='utf8')
    (OUT/'归档清单.json').write_text(json.dumps({'files':records,'models_or_data_copied':False,'original_reports_preserved':True},ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'归档核验.json').write_text(json.dumps({'source_hashes_unchanged':True,'file_count':len(records),'experiment_count':len(groups),'preexisting_unresolved_links':broken},ensure_ascii=False,indent=2),encoding='utf8')
    print(f'Archived {len(records)} report artifacts into {len(groups)} experiment folders; original files unchanged; {len(broken)} pre-existing unresolved links recorded.')

if __name__=='__main__':main()
