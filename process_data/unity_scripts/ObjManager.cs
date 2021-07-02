using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System;
using UnityEngine.UI;

public class objfile
{
    public string path { get; set; }    // 文件路径
    public int num { get; set; }    // 迭代次数
    public bool isManifold{ get; set; } // 重建流形
    public bool isFinale{ get; set; } // 最终效果

}
public class ObjManager : MonoBehaviour
{
    public GameObject mycamera;
    public GameObject myobj;
    public Vector3 camerapos;
    public Vector3 pos_g;
    public Vector3 pos_bull;
    public Vector3 pos_a;
    // public GameObject myobj_clone;
    public List<objfile> gofileArray;
    public int count=0;
    public int max_count;
    public int mytimer=0;

    // GUI 文本显示
    public bool flag_m=false;
    public bool flag_f=false;
    public int num_iter=0;
    public Text showInfo;
    public bool isDeform=false;
    // Start is called before the first frame update
    void Start()
    {
        count=0;
        max_count=0;
        showInfo.text="Iteration: "+count.ToString();
        mycamera.gameObject.transform.position=pos_g;
    }

    public void loadobj(string filepath){
        
        GameObject objPreb=Resources.Load(filepath) as GameObject;
        // print(objPreb);
        GameObject cloneobj=Instantiate(objPreb);
        MeshFilter[] meshes;
        meshes = cloneobj.gameObject.GetComponentsInChildren<MeshFilter>();
        myobj.GetComponent<MeshFilter>().mesh = meshes[0].mesh;
        // myobj_clone.GetComponent<MeshFilter>().mesh = meshes[0].mesh;
        Destroy(cloneobj);
        // Debug.Log(filepath);

    }

    // Update is called once per frame
    void Update()
    {
        if(mytimer%10==0){
            if(count<max_count){
                if(isDeform==false){
                    num_iter=gofileArray[count].num;
                    print(num_iter);
                    if(gofileArray[count].isManifold){
                        showInfo.text="Iteration: "+ num_iter.ToString() + " RWM Manifold";
                    }
                    else if(gofileArray[count].isFinale){
                        showInfo.text="Iteration: x"+ " Final Result";
                    }
                    else{
                        showInfo.text="Iteration: "+ num_iter.ToString();
                    }
                    loadobj(gofileArray[count].path);
                    count++;
                }
                else{
                    // 逆序
                    num_iter=gofileArray[max_count-1-count].num;
                    print(num_iter);
                    if(gofileArray[max_count-1-count].isManifold){
                        showInfo.text="Iteration: "+ num_iter.ToString() + " RWM Manifold";
                    }
                    else if(gofileArray[max_count-1-count].isFinale){
                        showInfo.text="Iteration: x"+ " Final Result";
                    }
                    else{
                        showInfo.text="Iteration: "+ num_iter.ToString();
                    }
                    loadobj(gofileArray[max_count-1-count].path);
                    count++;
                    if(count==max_count){
                        isDeform=false;
                        onCatButtonClick();
                    }
                }
                
            }
            
        }
        mytimer++;
        
        
    }
    public void AddFile2List(string folder_name){
        // 读取文件夹中所有的obj文件名字，存入list
        DirectoryInfo folder = new DirectoryInfo("./Assets/Resources/"+folder_name);
        var files = folder.GetFiles("*.obj");
        Debug.Log("files count :" + files.Length);
        gofileArray=new List<objfile>();
        count=0;
        max_count=0;
        
        for(int i = 0; i < files.Length; i++)
        {
            // Debug.Log(files[i].Name);
            string[] prefix=files[i].Name.Split('.');
            string[] file_pre=prefix[0].Split('_');

            if(file_pre[1]=="last"){
                gofileArray.Sort(delegate (objfile p1, objfile p2)
                {
                    return p1.num.CompareTo(p2.num);    // 排序
                });
                // foreach (var item in gofileArray)
                // {
                //     print(item.num);
                //     print("--path:"+item.path);
                // }
                gofileArray.Add(new objfile() { path = folder_name+"/"+prefix[0], num = 2000000, isManifold=false, isFinale=true });
            }
            else{
                int file_iter_num=0;
                bool Manifold=false;
                if(file_pre.Length<4){
                    file_iter_num=Convert.ToInt32(file_pre[2]);
                }
                else{
                    file_iter_num=Convert.ToInt32(file_pre[2])+1;
                    Manifold=true;
                }
                //print(file_iter_num);
                gofileArray.Add(new objfile() { path = folder_name+"/"+prefix[0], num = file_iter_num,isManifold=Manifold, isFinale=false });
            }
        }
        gofileArray.Sort(delegate (objfile p1, objfile p2)
        {
            return p1.num.CompareTo(p2.num);    // 排序
        });
        max_count=files.Length;
    }
    public void onDeformButtonClick(){
        isDeform=true;
        mycamera.gameObject.transform.position=pos_g;
        AddFile2List("giraffe");
    }
    public void onCatButtonClick(){
        DirectoryInfo folder = new DirectoryInfo("./Assets/Resources/cat");
        var files = folder.GetFiles("*.obj");
        Debug.Log("files count :" + files.Length);
        gofileArray=new List<objfile>();
        count=0;
        max_count=0;
        
        for(int i = 0; i < files.Length; i++)
        {
            // Debug.Log(files[i].Name);
            string[] prefix=files[i].Name.Split('.');
            string[] file_pre=prefix[0].Split('_');

            if(file_pre[1]=="last"){
                gofileArray.Sort(delegate (objfile p1, objfile p2)
                {
                    return p1.num.CompareTo(p2.num);    // 排序
                });

                gofileArray.Add(new objfile() { path = "cat/"+prefix[0], num = 2000000, isManifold=false, isFinale=true });
            }
            else{
                int file_iter_num=0;
                bool Manifold=false;
                if(file_pre.Length<4){
                    file_iter_num=Convert.ToInt32(file_pre[2]);
                }
                else{
                    file_iter_num=Convert.ToInt32(file_pre[2])+1;
                    Manifold=true;
                }
                //print(file_iter_num);
                gofileArray.Add(new objfile() { path = "cat/"+prefix[0], num = file_iter_num,isManifold=Manifold, isFinale=false });
            }
        }
        gofileArray.Sort(delegate (objfile p1, objfile p2)
        {
            return p1.num.CompareTo(p2.num);    // 排序
        });
        // 调整相机
        mycamera.gameObject.transform.position=camerapos;
        max_count=files.Length;
    }

    public void onGuitarButtonClick(){
        isDeform=false;
        mycamera.gameObject.transform.position=pos_g;
        AddFile2List("guitar");
    }

    public void onBullButtonClick(){
        isDeform=false;
        mycamera.gameObject.transform.position=pos_bull;
        AddFile2List("bull");
    }
    public void onGiraffeButtonClick(){
        isDeform=false;
        mycamera.gameObject.transform.position=pos_g;
        AddFile2List("giraffe");
    }
    public void onAmongusButtonClick(){
        isDeform=false;
        mycamera.gameObject.transform.position=pos_a;
        AddFile2List("amongus");
    }
    public void onAmongusNoiseButtonClick(){
        isDeform=false;
        mycamera.gameObject.transform.position=pos_a;
        AddFile2List("amongus_noise");
    }
}
