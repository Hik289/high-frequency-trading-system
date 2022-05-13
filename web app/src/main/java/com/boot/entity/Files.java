package com.boot.entity;

import java.io.Serializable;
import java.util.Date;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

import lombok.Data;
/**
 * files
 */
@Entity
@Table(name = "t_files")
@Data
public class Files implements Serializable{
	private static final long serialVersionUID = 874310660870093440L;
	@Id
	private String id;
	private String name;
	
	@Column(name="card_id")
	private Integer cardId;//
	
	private String path;
	
	@Column(name="create_time",updatable=false)
	private Date createTime = new Date();//创建时间
}
