using UnityEngine;
using UnityEditor;

public class ModelsModifier : MonoBehaviour
{
	private static string[] MyFolders = new string[] {"Assets/Resources/guitar" };
	[MenuItem("MyTools/ChangeModel")]
	public static void ChangeModel()
	{
		string[] guids = AssetDatabase.FindAssets("", MyFolders);
		Debug.Log(guids.Length);
		for (int i = 0; i < guids.Length; ++i)
		{
			string path = AssetDatabase.GUIDToAssetPath(guids[i]);
			Debug.Log(path);
			ModelImporter prefab = AssetDatabase.LoadAssetAtPath<ModelImporter>(path);
			Debug.Log(prefab);
			prefab.isReadable=true;
		}

		AssetDatabase.SaveAssets();
	}

}

public class DisableMaterialImport : AssetPostprocessor {
	void OnPreprocessModel ()
	{
		ModelImporter modelImporter = assetImporter as ModelImporter;
		modelImporter.isReadable=true;
	}
}
 